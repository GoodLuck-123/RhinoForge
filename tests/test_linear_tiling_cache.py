"""Production tile-cache equivalence, collision handling and thread isolation."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def selector_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    assert compiler, "tile-selector checks require a C++ compiler"
    directory = tmp_path_factory.mktemp("linear-tiling-cache")
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "src/ops/rpu_linear_tiling.h"
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <thread>
#include <iostream>
using namespace rpu_pl_tiling;
using detail::TileFamily;
constexpr TileFamily families[] = {
    TileFamily::Fp16Acc16, TileFamily::W8Acc16, TileFamily::Fp16Acc32,
    TileFamily::W8Acc32, TileFamily::W4Acc16, TileFamily::W4Acc32,
    TileFamily::Nvfp4Acc32};
struct Reference { const PLEntry* table; size_t count; int (*mtile)(int); };
Reference reference(TileFamily family) {
    switch (family) {
      case TileFamily::Fp16Acc16: return {PL_EXACT_W16, table_size(PL_EXACT_W16), mtile_w16};
      case TileFamily::W8Acc16: return {PL_EXACT_W8A16, table_size(PL_EXACT_W8A16), mtile_w8a16};
      case TileFamily::Fp16Acc32: return {PL_EXACT_FP16_ACC32, table_size(PL_EXACT_FP16_ACC32), mtile_fp16_acc32};
      case TileFamily::W8Acc32: return {PL_EXACT_W8A16_ACC32, table_size(PL_EXACT_W8A16_ACC32), mtile_w8a16_acc32};
      case TileFamily::W4Acc16: return {PL_EXACT_INT4, table_size(PL_EXACT_INT4), mtile_int4};
      case TileFamily::W4Acc32: return {PL_EXACT_INT4_ACC32, table_size(PL_EXACT_INT4_ACC32), mtile_int4_acc32};
      case TileFamily::Nvfp4Acc32: return {PL_EXACT_NVFP4_ACC32, table_size(PL_EXACT_NVFP4_ACC32), mtile_nvfp4_acc32};
    }
    std::abort();
}
TilePick uncached(TileFamily family, int m, int n, int k) {
    const auto ref = reference(family);
    int tile = exact_lookup(ref.table, ref.count, m, n, k);
    if (!tile) tile = measured_neighbor_tile(ref.table, ref.count, m, n, k, ref.mtile);
    return {tile, ref.mtile(tile)};
}
TilePick select(TileFamily family, int m, int n, int k) {
    switch (family) {
      case TileFamily::Fp16Acc16: return select_tile_acc16(m,n,k,true);
      case TileFamily::W8Acc16: return select_tile_acc16(m,n,k,false);
      case TileFamily::Fp16Acc32: return select_tile_acc32(m,n,k,true);
      case TileFamily::W8Acc32: return select_tile_acc32(m,n,k,false);
      case TileFamily::W4Acc16: return select_tile_int4(m,n,k);
      case TileFamily::W4Acc32: return select_tile_int4_acc32(m,n,k);
      case TileFamily::Nvfp4Acc32: return select_tile_nvfp4_acc32(m,n,k);
    }
    std::abort();
}
void same(TilePick a, TilePick b) {
    assert(a.n_tile == b.n_tile && a.m_tile == b.m_tile);
}
void equivalence() {
    // Geometry-specific choices must survive fallback scoring and cache reuse.
    const std::array<int,4> migrated[] = {
        {336,128,2048,64}, {336,256,2048,96}, {336,1376,2048,96},
        {336,2048,256,96}, {336,2048,1376,80}, {576,160,1280,96},
        {576,448,1280,96}, {576,1280,160,112}, {576,1280,448,96},
        {640,1376,2048,112},
    };
    for(const auto& shape:migrated)
        assert(select_tile_acc16(shape[0],shape[1],shape[2],false).n_tile==shape[3]);
    for (auto family : families) {
        const auto ref = reference(family);
        for (size_t i=0; i<ref.count; ++i) {
            const auto entry=ref.table[i];
            same(select(family,entry.M,entry.n,entry.k), {entry.tile,ref.mtile(entry.tile)});
            same(select(family,entry.M,entry.n,entry.k), {entry.tile,ref.mtile(entry.tile)});
        }
    }
    // Original public-selector results, captured before adding memoization.
    // Identical geometry must not mix dtype or accumulation families.
    const TilePick expected[]={{112,352},{112,320},{80,304},{80,288},
                               {128,304},{96,256},{64,320}};
    for(int repeat=0; repeat<3; ++repeat)
        for(size_t i=0; i<7; ++i) same(select(families[i],64,1536,4096),expected[i]);
    // Unmeasured shapes and tails exceed cache capacity. Revisit evicted
    // shapes in reverse order and compare to the original uncached algorithm.
    for(int direction : {1,-1}) for(int j=0; j<512; ++j) {
        const int i=direction==1?j:511-j;
        const int m=17+i%257,n=32+16*(i%73),k=128+64*(i%53);
        for(auto family : families) {
            const auto expected=uncached(family,m,n,k);
            same(select(family,m,n,k),expected);
            same(select(family,m,n,k),expected);
        }
    }
    for(auto family : families) for(const auto& shape :
            {std::array<int,3>{0,32,128}, {-1,32,128}, {64,0,128}, {64,32,-1}}) {
        bool original_rejected=false, cached_rejected=false;
        try { uncached(family,shape[0],shape[1],shape[2]); }
        catch(const std::invalid_argument&) { original_rejected=true; }
        try { select(family,shape[0],shape[1],shape[2]); }
        catch(const std::invalid_argument&) { cached_rejected=true; }
        assert(original_rejected && cached_rejected);
    }
}
void bounds() {
    detail::TileSelectionCache cache;
    static_assert(sizeof(cache)<=detail::TileSelectionCache::capacity*32);
    const int count=detail::TileSelectionCache::capacity*8;
    TilePick pick;
    assert(!cache.lookup(TileFamily::W4Acc16,64,1536,4096,pick));
    for(int i=0; i<count; ++i) {
        const auto family=families[i%7];
        cache.store(family,64+i,1536+i,4096+i,{i,i+1});
        assert(cache.lookup(family,64+i,1536+i,4096+i,pick));
        same(pick,{i,i+1});
    }
    size_t retained=0;
    for(int i=0; i<count; ++i) {
        if(cache.lookup(families[i%7],64+i,1536+i,4096+i,pick)) {
            ++retained; same(pick,{i,i+1});
        }
    }
    assert(retained>0 && retained<=detail::TileSelectionCache::capacity && retained<count);
    // Values that collide under the exact-table packed key are distinct memo
    // identities. A direct-mapped slot may evict, but must never return aliases.
    assert(pl_key(64,1537,4096)==pl_key(64,1536,4096+(1<<17)));
    for(int k : {4096,4096+(1<<17)}) for(int n : {1536,1537})
        for(int m : {64,65}) for(auto family : families) {
            cache.store(family,m,n,k,{m+n+1000*static_cast<int>(family),k});
            for(int other_k : {4096,4096+(1<<17)}) for(int other_n : {1536,1537})
                for(int other_m : {64,65}) for(auto other_family : families)
                    if(cache.lookup(other_family,other_m,other_n,other_k,pick))
                        same(pick,{other_m+other_n+1000*static_cast<int>(other_family),other_k});
        }
    int attempts=0;
    for(int i=0;i<2;++i) {
        bool rejected=false;
        try {
            detail::cached_tile(TileFamily::W4Acc16,-17,123,456,[&]() -> TilePick {
                ++attempts; throw std::invalid_argument("failed computation");
            });
        } catch(const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
    assert(attempts==2);
    // The memo key preserves signed int boundaries; hashing must not fold
    // distinct fields even when their unsigned representations are extreme.
    // These direct cache probes do not admit invalid selector geometries.
    const int extremes[]={std::numeric_limits<int>::min(),-1,0,1,
                          std::numeric_limits<int>::max()};
    for(int m=0;m<5;++m) for(int n=0;n<5;++n) for(int k=0;k<5;++k) {
        cache.store(TileFamily::Fp16Acc16,extremes[m],extremes[n],extremes[k],
                    {25*m+5*n+k,0});
        for(int om=0;om<5;++om) for(int on=0;on<5;++on) for(int ok=0;ok<5;++ok)
            if(cache.lookup(TileFamily::Fp16Acc16,extremes[om],extremes[on],extremes[ok],pick))
                same(pick,{25*om+5*on+ok,0});
    }
}
void concurrent() {
    std::atomic<int> arrived{0};
    std::array<const detail::TileSelectionCache*,4> addresses{};
    std::vector<std::thread> threads;
    for(int thread=0;thread<4;++thread) threads.emplace_back([&,thread] {
        addresses[thread]=&detail::tile_selection_cache();
        ++arrived;
        while(arrived.load()!=4) std::this_thread::yield();
        for(int i=0;i<96;++i) for(auto family:families) {
            const int m=64+(i+thread)%37,n=32+16*(i%33),k=128+64*(i%17);
            same(select(family,m,n,k),uncached(family,m,n,k));
        }
    });
    for(auto& thread:threads) thread.join();
    for(size_t i=0;i<addresses.size();++i) for(size_t j=i+1;j<addresses.size();++j)
        assert(addresses[i]!=addresses[j]);
}
int main(int argc,char** argv) {
    assert(argc==2);
    const std::string mode=argv[1];
    if(mode=="equivalence") equivalence();
    else if(mode=="bounds") bounds();
    else if(mode=="threads") concurrent();
    else return 2;
    std::cout<<mode<<" passed\n";
}
''')
    executable = directory / "probe"
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-I", str(ROOT),
         str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    return executable


@pytest.mark.parametrize("mode", ["equivalence", "bounds", "threads"])
def test_tile_selection_cache(selector_probe, mode):
    result = subprocess.run([str(selector_probe), mode],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
