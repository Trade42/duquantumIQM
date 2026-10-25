/* Nanosecond decoder for the compiled GPU ensemble: lookup table + MLP fallback, with a latency benchmark.
 *
 * Inputs (little-endian, written by iyanu/latency/prepare.py):
 *   table.bin  "DQLT", u32 mode (0 dense, 1 sparse), u32 n_det, u64 entries, u32 n_layers,
 *              dense: (2^n_det)/8 bytes of bits | sparse: u32 keys[entries] (sorted), u8 bits[entries],
 *              then per MLP layer: u32 rows, u32 cols, f32 W[rows*cols], f32 b[rows]
 *   shots.bin  "DQSH", u64 n, u32 keys[n], u8 obs[n]    (held-out test syndromes and true flips)
 *
 * decode(key): dense -> one bit read; sparse -> open-addressing hash probe, miss -> MLP forward.
 * Timing: (a) streaming: every test shot once, in order, total time / n; (b) per shot: each shot
 * decoded REPS times as a dependency chain (each key depends on the previous answer), per-call time = elapsed/REPS, giving a per-syndrome latency
 * distribution below the clock tick (macOS mach_absolute_time ticks are ~41.7 ns).
 * Errors are counted against the true flips, to check the C decoder matches the Python one.
 *
 *   cc -O3 -march=native -o decode_bench decode_bench.c && ./decode_bench table.bin shots.bin
 */
#include <stdint.h>
#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>   /* cblas_sgemv on the Apple matrix units */
#else
#include <cblas.h>
#endif
#if defined(__APPLE__)
#include <mach/mach_time.h>
static double ns_per_tick;
static void clock_init(void) { mach_timebase_info_data_t t; mach_timebase_info(&t); ns_per_tick = (double)t.numer / t.denom; }
static inline uint64_t now(void) { return mach_absolute_time(); }
#else
#include <time.h>
static double ns_per_tick = 1.0;
static void clock_init(void) {}
static inline uint64_t now(void) { struct timespec s; clock_gettime(CLOCK_MONOTONIC_RAW, &s); return (uint64_t)s.tv_sec*1000000000ull + s.tv_nsec; }
#endif

#define REPS 256
static volatile uint32_t opaque_zero = 0;  /* defeats hoisting: every repetition is a real lookup */

typedef struct { uint32_t rows, cols; float *w, *b; } Layer;
static uint32_t mode, n_det, n_layers;
static uint64_t entries;
static uint8_t *dense;
static uint64_t *slots;  /* sparse: hash slots holding key+1 (0 = empty) in the low 33 bits, answer in bit 40 */
static uint64_t slot_mask;
static Layer layers[8];
static float buf_a[4096], buf_b[4096];
static uint64_t fallbacks;

static void die(const char *m) { fprintf(stderr, "error: %s\n", m); exit(1); }
static void rd(void *p, size_t n, FILE *f) { if (fread(p, 1, n, f) != n) die("short read"); }

static inline uint64_t hash(uint32_t k) { return ((uint64_t)k * 0x9E3779B97F4A7C15ull) >> 20; }

static void load_table(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("open table");
    char magic[4]; rd(magic, 4, f); if (memcmp(magic, "DQLT", 4)) die("bad table magic");
    rd(&mode, 4, f); rd(&n_det, 4, f); rd(&entries, 8, f); rd(&n_layers, 4, f);
    if (n_layers > 8) die("too many layers");
    if (mode == 0) {
        size_t bytes = ((size_t)1 << n_det) / 8 + 1;
        dense = malloc(bytes); rd(dense, ((size_t)1 << n_det) / 8 ? ((size_t)1 << n_det) / 8 : 1, f);
    } else {
        uint32_t *keys = malloc(entries * 4); uint8_t *bits = malloc(entries);
        rd(keys, entries * 4, f); rd(bits, entries, f);
        uint64_t size = 1; while (size < 2 * entries) size <<= 1;
        slots = calloc(size, 8); slot_mask = size - 1;
        for (uint64_t i = 0; i < entries; i++) {
            uint64_t h = hash(keys[i]) & slot_mask;
            while (slots[h]) h = (h + 1) & slot_mask;
            slots[h] = ((uint64_t)keys[i] + 1) | ((uint64_t)(bits[i] & 1) << 40);
        }
        free(keys); free(bits);
    }
    for (uint32_t l = 0; l < n_layers; l++) {
        rd(&layers[l].rows, 4, f); rd(&layers[l].cols, 4, f);
        if (layers[l].rows > 4096 || layers[l].cols > 4096) die("layer too wide");
        layers[l].w = malloc((size_t)layers[l].rows * layers[l].cols * 4); layers[l].b = malloc(layers[l].rows * 4);
        rd(layers[l].w, (size_t)layers[l].rows * layers[l].cols * 4, f); rd(layers[l].b, layers[l].rows * 4, f);
    }
    fclose(f);
}

static float *w0t;  /* first layer transposed: column j of W0 is contiguous */

static void prepare_mlp(void) {
    const Layer *L = &layers[0];
    w0t = malloc((size_t)L->rows * L->cols * 4);
    for (uint32_t r = 0; r < L->rows; r++)
        for (uint32_t c = 0; c < L->cols; c++) w0t[(size_t)c * L->rows + r] = L->w[(size_t)r * L->cols + c];
}

static uint8_t mlp(uint32_t key) {
    fallbacks++;
    const Layer *L0 = &layers[0];
    float *x = buf_a, *y = buf_b;
    memcpy(x, L0->b, L0->rows * 4);                  /* binary input: layer 0 = bias + columns of fired detectors */
    for (uint32_t m = key; m; m &= m - 1) {
        const float *col = w0t + (size_t)__builtin_ctz(m) * L0->rows;
        for (uint32_t r = 0; r < L0->rows; r++) x[r] += col[r];
    }
    for (uint32_t r = 0; r < L0->rows; r++) if (x[r] < 0) x[r] = 0;
    for (uint32_t l = 1; l < n_layers; l++) {
        const Layer *L = &layers[l];
        memcpy(y, L->b, L->rows * 4);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)L->rows, (int)L->cols, 1.0f, L->w, (int)L->cols, x, 1, 1.0f, y, 1);
        if (l + 1 < n_layers) for (uint32_t r = 0; r < L->rows; r++) if (y[r] < 0) y[r] = 0;
        float *t = x; x = y; y = t;
    }
    return x[0] > 0;
}

static inline uint8_t decode(uint32_t key) {
    if (mode == 0) return (dense[key >> 3] >> (key & 7)) & 1;
    uint64_t h = hash(key) & slot_mask, want = (uint64_t)key + 1;
    for (;;) {
        uint64_t s = slots[h];
        if (!s) return mlp(key);
        if ((s & 0x1FFFFFFFFull) == want) return (s >> 40) & 1;
        h = (h + 1) & slot_mask;
    }
}

static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s table.bin shots.bin\n", argv[0]); return 2; }
    clock_init();
#if defined(__APPLE__)
    fesetenv(FE_DFL_DISABLE_DENORMS_ENV);      /* flush denormals to zero: they stall the matrix units */
#endif
    load_table(argv[1]);
    prepare_mlp();
    FILE *f = fopen(argv[2], "rb"); if (!f) die("open shots");
    char magic[4]; rd(magic, 4, f); if (memcmp(magic, "DQSH", 4)) die("bad shots magic");
    uint64_t n; rd(&n, 8, f);
    uint32_t *keys = malloc(n * 4); uint8_t *obs = malloc(n);
    rd(keys, n * 4, f); rd(obs, n, f); fclose(f);

    if (getenv("BENCH_MLP")) {                   /* time the fallback alone, and each layer */
        uint64_t a = now(); volatile uint8_t v = 0;
        for (int r = 0; r < 2000; r++) v += mlp(keys[r % n]);
        fprintf(stderr, "mlp(): %.0f ns/call over 2000 calls (layers:", (now() - a) * ns_per_tick / 2000);
        for (uint32_t l = 0; l < n_layers; l++) fprintf(stderr, " %ux%u", layers[l].rows, layers[l].cols);
        fprintf(stderr, ")\n");
        a = now(); for (int r = 0; r < 2000; r++) v += mlp(keys[0]);
        fprintf(stderr, "mlp(same key x2000): %.0f ns/call\n", (now() - a) * ns_per_tick / 2000);
        a = now(); for (int r = 0; r < 2000; r++) v += decode(keys[r % n]);
        fprintf(stderr, "decode(varied) incl. hits: %.0f ns/call, fallbacks %llu\n", (now() - a) * ns_per_tick / 2000, (unsigned long long)fallbacks);
        fallbacks = 0;
    }
    volatile uint64_t sink = 0; uint64_t errors = 0;
    for (uint64_t i = 0; i < n; i++) errors += decode(keys[i]) != obs[i];      /* accuracy + warm-up */
    uint64_t fb_stream = fallbacks; fallbacks = 0;

    uint64_t t0 = now();                                                       /* (a) streaming */
    for (uint64_t i = 0; i < n; i++) sink += decode(keys[i]);
    double stream_ns = (now() - t0) * ns_per_tick / n;

    double *per = malloc(n * sizeof(double));                                  /* (b) per shot */
    for (uint64_t i = 0; i < n; i++) {
        uint32_t k = keys[i], z = opaque_zero, prev = 0; uint64_t a = now();
        for (int r = 0; r < REPS; r++) prev = decode(k ^ (prev & z));      /* serial dependency chain */
        sink += prev;
        per[i] = (now() - a) * ns_per_tick / REPS;
    }
    qsort(per, n, sizeof(double), cmp);
    #define Q(p) per[(uint64_t)((p) * (n - 1))]
    printf("{\"mode\": \"%s\", \"n_det\": %u, \"entries\": %llu, \"test_shots\": %llu, \"errors\": %llu, "
           "\"fallback_shots\": %llu, \"stream_mean_ns\": %.2f, \"p50_ns\": %.2f, \"p90_ns\": %.2f, \"p99_ns\": %.2f, "
           "\"p999_ns\": %.2f, \"max_ns\": %.2f, \"reps_per_shot\": %d}\n",
           mode ? "sparse" : "dense", n_det, (unsigned long long)entries, (unsigned long long)n, (unsigned long long)errors,
           (unsigned long long)fb_stream, stream_ns, Q(0.5), Q(0.9), Q(0.99), Q(0.999), per[n - 1], REPS);
    return (int)(sink & 0);
}
