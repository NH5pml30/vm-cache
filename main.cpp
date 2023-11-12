#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <memory>
#include <span>

#include <sched.h>

constexpr size_t MEM_ALIGNMENT = (1 << 10) * 32; // 32K
constexpr size_t MIN_CACHE_LINE = 16;            // 16 bytes

void set_affinity(int proc_i) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(proc_i, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

auto alloc_aligned(size_t size) {
  return std::unique_ptr<std::byte, void (*)(std::byte *)>(
      static_cast<std::byte *>(std::aligned_alloc(MEM_ALIGNMENT, size)),
      [](std::byte *ptr) { std::free(ptr); });
}

__attribute__((noinline)) void reset_cache(volatile std::byte *ptr,
                                           size_t size) {
  for (size_t i = 0; i < size; i += MIN_CACHE_LINE)
    ptr[i] = std::byte(0);
}

void shuffle(std::span<uintptr_t> span, size_t stride, auto &random_engine) {
  // Sattolo's algorithm
  for (size_t i = 0; i < span.size() / stride; i++)
    span[i * stride] = reinterpret_cast<uintptr_t>(&span[i * stride]);

  size_t i = span.size() / stride;
  while (i > 1) {
    i--;
    size_t j = std::uniform_int_distribution<size_t>(0, i - 1)(random_engine);
    std::swap(span[i * stride], span[j * stride]);
  }
}

using rep_t = std::chrono::nanoseconds::rep;
rep_t measure(std::span<uintptr_t> span, size_t n_iters);

using rep_t = std::chrono::nanoseconds::rep;

void loop(uintptr_t *start_ptr, size_t n_iters) {
  // do not unroll the loop
#pragma GCC unroll 1
  for (size_t i = 0; i < n_iters; i++) {
    volatile uintptr_t *ptr = start_ptr;
    do {
      ptr = reinterpret_cast<volatile uintptr_t *>(*ptr);
    } while (ptr != start_ptr);
  }
}

__attribute__((noinline)) rep_t measure(std::span<uintptr_t> span,
                                        size_t n_iters) {
  uintptr_t *start_ptr = span.data();
  auto start = std::chrono::high_resolution_clock::now();

  loop(start_ptr, n_iters);

  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
      .count();
}

auto measure_ws_assoc(std::span<std::byte> data, size_t waysize, size_t assoc,
                      size_t n_sum_iters, size_t n_min_iters, auto &rand_eng) {
  assert(data.size() >= assoc * waysize);

  // select the minimum time
  rep_t min = std::numeric_limits<long long>::max();
  for (size_t i = 0; i < n_min_iters; i++) {
    auto span = std::span(reinterpret_cast<uintptr_t *>(data.data()),
                          assoc * waysize / sizeof(uintptr_t));
    shuffle(span, waysize / sizeof(uintptr_t), rand_eng);
    min = std::min(min, measure(span, n_sum_iters));
  }
  return min * 1.0 / assoc;
}

using cache_info_t = std::pair<size_t, size_t>;

bool is_pow_of_2(size_t x) { return (x & (x - 1)) == 0; }

cache_info_t detect(size_t max_waysize, size_t max_assoc, double jump_rel,
                    size_t n_sum_iters, size_t n_min_iters) {
  assert(is_pow_of_2(max_waysize));
  auto eng = std::default_random_engine{};
  size_t max_mem = max_waysize * max_assoc;
  auto data = alloc_aligned(max_mem);
  reset_cache(data.get(), max_mem);

  std::vector<ptrdiff_t> last_jumps, last_ws;
  // assume that waysize is a power of 2
  size_t ws = max_waysize;
  while (ws >= MIN_CACHE_LINE) {
    std::cout << "Ws = " << ws << "\n";
    double sum = 0;
    size_t n_measurements = 0;
    ptrdiff_t jump = -1;
    for (size_t assoc = 1; assoc < max_assoc; assoc++) {
      std::cout << "  Assoc = " << assoc << "\n";
      auto time = measure_ws_assoc(std::span(data.get(), max_mem), ws, assoc,
                                   n_sum_iters, n_min_iters, eng);
      double avg = sum / n_measurements;
      double rel = time / avg;

      std::cout << "    time = " << time << " / " << avg << " = " << rel
                << "\n";
      if (n_measurements > 0 && rel > jump_rel) {
        std::cout << "-> jump\n";
        jump = assoc - 1;
        break;
      }
      sum += time;
      n_measurements++;
    }

    if (!last_jumps.empty() && last_jumps.back() >= 0 && jump != last_jumps.back()) {
      if (jump == last_jumps.back() * 2)
        break;
      if (jump == (last_jumps.back() - 1) * 2) {
        last_jumps.back()--;
        std::cout << "assuming last jump was late, returning to " << last_jumps.back() << "\n";
        break;
      }

      std::cout << "-> rollback\n";
      last_jumps.pop_back();
      last_ws.pop_back();
      ws *= 2;
      jump_rel *= 0.97;
      continue;
    }
    last_jumps.push_back(jump);
    last_ws.push_back(ws);
    ws /= 2;
  }
  if (last_jumps.empty())
    return {-1, -1};
  return {last_jumps.back() * last_ws.back(), last_jumps.back()};
}

template <typename T> std::optional<T> vote(const std::vector<T> &vals) {
  for (size_t i = 0; i < vals.size(); i++) {
    size_t matched = 0;
    for (size_t j = 0; j < vals.size(); j++)
      if (i != j)
        matched += vals[i] == vals[j];

    if (matched + 1 > vals.size() / 2)
      return vals[i];
  }

  return {};
}

int main() {
  constexpr int CORE_IDX = 0;
  constexpr size_t MAX_WAYSIZE = (size_t{1} << 10) * 16; // 16K
  constexpr size_t MAX_ASSOC = 32;
  constexpr double JUMP_REL = 1.35;
  constexpr size_t N_SUM_ITERS = 50'000;
  constexpr size_t N_MIN_ITERS = 1'000;
  constexpr size_t N_VOTES = 3;

  set_affinity(CORE_IDX);

  std::vector<cache_info_t> infos;
  std::generate_n(std::back_inserter(infos), N_VOTES, []() {
    return detect(MAX_WAYSIZE, MAX_ASSOC, JUMP_REL, N_SUM_ITERS, N_MIN_ITERS);
  });

  for (auto info : infos)
    std::cout << "[" << info.first << ", " << info.second << "]\n";

  auto result = vote(infos);
  if (!result.has_value() || result->first == -1) {
    std::cout << "failed" << std::endl;
    return 1;
  }

  auto [cache_size, cache_assoc] = *result;
  std::cout << cache_size << " bytes for core " << CORE_IDX << ", "
            << cache_assoc << "-way set associative" << std::endl;
  return 0;
}
