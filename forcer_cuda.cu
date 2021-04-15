#include <cstdint>
#include <cassert>
#include <chrono>
#include <fmt/format.h>

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

struct ForcerContext {
	u8 rng_state[4];
	u8 work_buffer[8*8];
	u8 decompress_buffer[0x30];
	u8 preloaded_map[8*8*9];
	u8* found_byte;
};

static const unsigned seed_count = 0xfffffff + 1;
static const unsigned block_count = 32;
static const unsigned threads_per_block = 1024;
static const unsigned core_count = block_count * threads_per_block;
static const unsigned seeds_per_thread = seed_count / core_count;
static const unsigned context_size = sizeof(ForcerContext) * core_count;


__device__ u8 rng_next(ForcerContext* ctx) {
	if(!ctx) return 0x0;

	ctx->rng_state[0]++;
	ctx->rng_state[1] = ctx->rng_state[3] ^ ctx->rng_state[0] ^ ctx->rng_state[1];
	ctx->rng_state[2] = ctx->rng_state[1] + ctx->rng_state[2];
	ctx->rng_state[3] = ctx->rng_state[3] + ((ctx->rng_state[2]>>1) ^ ctx->rng_state[1]);

	return ctx->rng_state[3];
}

__device__ void rng_reinitialize(ForcerContext* ctx, u32 seed) {
	if(!ctx) return;

	ctx->rng_state[3] = seed & 0xffu;
	ctx->rng_state[2] = (seed >> 8u) & 0xffu;
	ctx->rng_state[1] = (seed >> 16u) & 0xffu;
	ctx->rng_state[0] = (seed >> 24u) & 0xffu;

	for(unsigned i = 0; i < 0x10; ++i)
		rng_next(ctx);
}

__device__ u8 gen_ctl(ForcerContext* ctx, u32 seed, u16 x, u16 y) {
	if(!ctx) return 0x0;

	u8 masked_x = x & 3,
	   masked_y = y & 3;

	u32 mask = ((u32)(y & 0xFFFC) << 16u) | (x & 0xFFFC);
	rng_reinitialize(ctx, seed ^ mask);
	u8 b1 = rng_next(ctx) & 7;

	static const u8 dd0c_lookup[128] = {
			0x05,0x0B,0x06,0x00,0x0E,0x05,0x0B,0x03,0x09,0x0E,0x00,0x00,0x00,0x0C,0x00,0x00,0x00,0x0C,0x05,0x06,0x06,0x0D,0x0A,0x09,0x09,0x0B,0x06,0x00,0x00,0x05,0x0A,0x00,0x05,0x0B,0x06,0x00,0x0B,0x06,0x09,0x07,0x00,0x09,0x07,0x0A,0x00,0x05,0x0A,0x00,0x00,0x09,0x06,0x00,0x03,0x06,0x0C,0x05,0x00,0x09,0x0F,0x0A,0x00,0x05,0x0A,0x00,0x05,0x0A,0x00,0x00,0x0F,0x06,0x05,0x03,0x09,0x0F,0x0A,0x00,0x00,0x0C,0x00,0x00,0x05,0x0B,0x03,0x06,0x0E,0x00,0x00,0x0D,0x09,0x06,0x00,0x0C,0x00,0x0D,0x03,0x0A,0x00,0x09,0x06,0x00,0x06,0x05,0x0A,0x05,0x09,0x0F,0x06,0x0C,0x00,0x0C,0x09,0x0A,0x00,0x0C,0x00,0x00,0x03,0x0F,0x07,0x03,0x00,0x0D,0x0A,0x00,0x00,0x0C,0x00,0x00
	};

	auto offset = 4 * masked_y + masked_x;
	auto table_offset = offset + (b1 << 4u);

	u8 lookup_value = dd0c_lookup[table_offset];
	u8 b2 = rng_next(ctx) & 0x30;
	b2 |= lookup_value;

	auto result = (((x&0xFF) | (y&0xFF)) & 0xFC) | ((x>>8u) | (y>>8u));
	if(result != 0)
		return b2;
	else
		return b2 & 0x0F;
}

__device__ void map_decompress(ForcerContext* ctx, u8 new_tile) {
	if(!ctx) return;

	//  Copy tiles to not trample over things while modifying
	memcpy(&ctx->decompress_buffer[0], &ctx->work_buffer[8], 0x30);

	for(unsigned i = 8; i < 8*7; ++i) {
		if(ctx->decompress_buffer[i - 8] != new_tile)
			continue;
		if(((i&7) == 0) || ((i&7) == 7))
			continue;

		u8 v = rng_next(ctx);
		if(v & 1) {
			ctx->work_buffer[i-1] = new_tile;
		}
		if(v & 2) {
			ctx->work_buffer[i+1] = new_tile;
		}
		if(v & 4) {
			ctx->work_buffer[i-8] = new_tile;
		}
		if(v & 8) {
			ctx->work_buffer[i+8] = new_tile;
		}
	}
}

__device__ void map_place_tile_prob(ForcerContext* ctx, u8 old, u8 new_tile, u8 threshold) {
	if(!ctx) return;

	for(unsigned i = 0; i < 8*8; ++i) {
		if(ctx->work_buffer[i] != old)
			continue;
		u8 v = rng_next(ctx);
		if(v >= threshold)
			continue;

		ctx->work_buffer[i] = new_tile;
	}
}

//  Same as above, but skips lines 0 and 7 and tiles 0,7 on each line
//  presumably to avoid softlocking
__device__ void map_place_tile_prob_safe(ForcerContext* ctx, u8 old, u8 new_tile, u8 threshold) {
	if(!ctx) return;

	for(unsigned i = 8; i < 8*7; ++i) {
		if((i&7) == 0 || (i&7) == 7)
			continue;
		if(ctx->work_buffer[i] != old)
			continue;
		u8 v = rng_next(ctx);
		if(v >= threshold)
			continue;

		ctx->work_buffer[i] = new_tile;
	}
}

__device__ void map_place_tile_at(ForcerContext* ctx, u8 tile, u8 x, u8 y) {
	x = x & 0x0f;
	y = y & 0x0f;
	ctx->work_buffer[y * 8 + x] = tile;
}

__device__ void map_place_line_impl(ForcerContext* ctx, u8 tile, u8 pos1, u8 pos2) {
	if(!ctx) return;

	u8 x1 = (pos1 >> 4u) & 0xf,
			x2 = (pos2 >> 4u) & 0xf,
			y1 = pos1 & 0x0f,
			y2 = pos2 & 0x0f;

	u8 ystep = (y1 < y2) ? 0x01 : 0xFF,
			xstep = (x1 < x2) ? 0x01 : 0xFF;
	while((x1 != x2) || (y1 != y2)) {
		map_place_tile_at(ctx, tile, x1, y1);
		if(x1 != x2)
			x1 += xstep;
		map_place_tile_at(ctx, tile, x1, y1);
		if(y1 != y2)
			y1 += ystep;
		map_place_tile_at(ctx, tile, x1, y1);
	}
}

__device__ void map_place_line(ForcerContext* ctx, u8 tile, u8 start, u8 end) {
	if(!ctx) return;

	u8 x = rng_next(ctx) & 7;
	while(x == 0 || x == 7)
		x = rng_next(ctx) & 7;
	u8 y = rng_next(ctx) & 7;
	while(y == 0 || y == 7)
		y = rng_next(ctx) & 7;

	u8 point_pos = (x << 4u) | y;

	map_place_line_impl(ctx, tile, start, point_pos);
	map_place_line_impl(ctx, tile, point_pos, end);
}


__device__ void maybe_map_gen(ForcerContext* ctx, u8 old, u8 new_tile, u8 threshold, u8 hfindpathflags, u8 hmultiplier, u8 hfindpathxprogress, u8 hmultiplybuffer) {
	if(!ctx) return;

	//  b - old tile
	//  c - new tile
	//  d - threshold

	//  ???? - findpathflags
	//  ???? - multiplier
	//  ???? - findpathxprogress
	//  ???? - multiplybuffer

	//  tile - hmutatewx
	unsigned i = 8, c = 0x30;
	while(c > 0) {
		if((i&7) == 0 || (i&7) == 7) {
			i++;
			c--;
			continue;
		}
		if(ctx->work_buffer[i] != old) {
			i++;
			c--;
			continue;
		}

		u8 v = rng_next(ctx);
		if(v < threshold) {
			i++;
			c--;
			continue;
		}

		i -= 0x8;
		if(hfindpathflags != 0) {
			if(ctx->work_buffer[i] != hfindpathflags) {
				i += 0x9;
				c--;
				continue;
			}
		}

		i += 0x10;
		if(hmultiplier != 0) {
			if(ctx->work_buffer[i] != hmultiplier) {
				i -= 0x7;
				c--;
				continue;
			}
		}

		i -= 0x09;
		if(hfindpathxprogress != 0) {
			if(ctx->work_buffer[i] != hfindpathxprogress) {
				i += 2;
				c--;
				continue;
			}
		}

		i += 2;
		if(hmultiplybuffer != 0) {
			if(ctx->work_buffer[i] != hmultiplybuffer) {
				c--;
				continue;
			}
		}

		ctx->work_buffer[i-1] = new_tile;
		c--;
	}
}


__device__ void gen_map_data(ForcerContext* ctx, u32 seed, u16 x, u16 y) {
	if(!ctx) return;
	//fmt::print("Generating map[{},{}] with seed={:08x}\n", x, y, seed);

	//  Generate control byte
	auto ctl = gen_ctl(ctx, seed, x,y);

	//  Fill work buffer with 0F
	memset(&ctx->work_buffer[0], 0x0F, 8 * 8);

	u8 var1 = 0;
	if(ctl & 1)
		var1 = 0x74;
	if(ctl & 2)
		var1 = 0x04;
	if(ctl & 4)
		var1 = 0x47;
	if(ctl & 8)
		var1 = 0x40;

	//  Reinitialize RNG
	u32 seed_mask = ((u32)x << 16u) | y;
	rng_reinitialize(ctx, seed ^ seed_mask);


	//  World gen?
	if(ctl & 1)
		map_place_line(ctx, 0xA, var1, 0x74);
	if(ctl & 2)
		map_place_line(ctx, 0xA, var1, 0x04);
	if(ctl & 4)
		map_place_line(ctx, 0xA, var1, 0x47);
	if(ctl & 8)
		map_place_line(ctx, 0xA, var1, 0x40);


	//  "Decompression"?
	map_decompress(ctx, 0x0A);


	//  Place exits
	if(ctl & 8) {
		map_place_tile_at(ctx, 0xa, 0x3, 0x0);
		map_place_tile_at(ctx, 0xa, 0x4, 0x0);
	}
	if(ctl & 4) {
		map_place_tile_at(ctx, 0xa, 0x3, 0x7);
		map_place_tile_at(ctx, 0xa, 0x4, 0x7);
	}
	if(ctl & 2) {
		map_place_tile_at(ctx, 0xa, 0x0, 0x3);
		map_place_tile_at(ctx, 0xa, 0x0, 0x4);
	}
	if(ctl & 1) {
		map_place_tile_at(ctx, 0xa, 0x7, 0x3);
		map_place_tile_at(ctx, 0xa, 0x7, 0x4);
	}

	//  Biome specific generation
	auto biome_ctl = (ctl >> 4u) & 0x3;
	switch(biome_ctl) {
		case 0: {
			map_place_tile_prob(ctx, 0xa, 0xb, 0x30);
			map_decompress(ctx, 0x0B);
			maybe_map_gen(ctx, 0x0f, 0x6c, 0x20, 0x0f, 0x0a, 0x0, 0x0);
			maybe_map_gen(ctx, 0x0f, 0x6f, 0x20, 0x0a, 0x0f, 0x0, 0x0);
			maybe_map_gen(ctx, 0x0f, 0x6e, 0x20, 0x0, 0x0, 0x0a, 0x0f);
			maybe_map_gen(ctx, 0x0f, 0x6d, 0x20, 0x0, 0x0, 0x0f, 0x0a);
			map_place_tile_prob(ctx, 0xa, 0x74, 0x30);
			map_place_tile_prob(ctx, 0xa, 0x7a, 0x30);
			map_place_tile_prob_safe(ctx, 0x6c, 0x33, 0x40);
			map_place_tile_prob_safe(ctx, 0x6d, 0x32, 0x40);
			map_place_tile_prob_safe(ctx, 0x6e, 0x60, 0x40);
			map_place_tile_prob_safe(ctx, 0x6f, 0x34, 0x40);

			break;
		}
		case 1: {
			map_place_tile_prob(ctx, 0xa, 0x7b, 0x40);
			map_place_tile_prob(ctx, 0xa, 0x7a, 0x30);
			map_place_tile_prob(ctx, 0xa, 0xb, 0xd0);
			map_place_tile_prob_safe(ctx, 0x0a, 0x08, 0x20);
			break;
		}
		case 2: {
			//  Screw this
			assert(false);
			break;
		}
		case 3: {
			//  Screw this
			assert(false);
			break;
		}
		default: break;
	}
}

__device__ void generate_visible_map(ForcerContext* ctx, u32 seed) {
	if(!ctx) return;

	auto get_base = [](u8 x, u8 y) {
		if(y == 0) {
			return 0x0 + x*8;
		} else if(y == 1) {
			return 0xC0 + x*8;
		} else {
			return 0x180 + x*8;
		}
	};

	for(unsigned x = 0; x < 3; ++x) {
		for(unsigned y = 0; y < 3; ++y) {
			gen_map_data(ctx, seed, x, y+1);
			auto base = get_base(x,y);

			//  Copy lines such that the lines in nearby chunks are sequential in memory
			for(unsigned i = 0; i < 8; ++i) memcpy(&ctx->preloaded_map[base + i * 0x18], &ctx->work_buffer[i * 8], 8);
		}
	}
}

__device__ bool search_sequence(ForcerContext* ctx, unsigned windowx, unsigned windowy) {
	if(!ctx) return false;

	if(windowx >= 24 || windowx + 5 >= 24 || windowy >= 24 || windowy + 5 >= 24)
		return false;

	const u8 sequence[25] = {
			0x0b,0x0b,0x0b,0x74,0x0a,
			0x0f,0x0b,0x0f,0x0a,0x0a,
			0x0f,0x0f,0x0a,0x0a,0x0b,
			0x0f,0x0b,0x0a,0x0a,0x0a,
			0x0b,0x0b,0x0a,0x0a,0x74
	};

	unsigned c = 0;
	for(unsigned y = windowy; y < windowy + 5; ++y) {
		for(unsigned x = windowx; x < windowx + 5; ++x) {
			auto addr = y * 24 + x;
			if(ctx->preloaded_map[addr] != sequence[c])
				return false;
			c++;
			if(c == 25)
				return true;
		}
	}

	return false;
}

__global__ void forcer_entrypoint(void* context_pool_base, u32 cycle_base) {
	if(!context_pool_base)
		return;

	auto threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	auto blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
	auto threadsPerBlock  = blockDim.x * blockDim.y;
	auto thread_number = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	auto* ctx = (ForcerContext*)((u8*)context_pool_base + thread_number * sizeof(ForcerContext));

	u32 input_seed = cycle_base + thread_number;
	auto seed = (input_seed << 4u) | 0x01u;
	generate_visible_map(ctx, seed);

	//  Search the possible window for the sequence
	for(unsigned x = 6; x <= 13; ++x) {
		for(unsigned y = 6; y <= 13; ++y) {
			bool res = search_sequence(ctx, x,y);
			if(res) {
				printf("[Thread %d] SEED=%08x Found pattern occurence!\n", thread_number, seed);
				return;
			}
		}
	}
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int main() {
	fmt::print("Bruteforce using {} blocks, {} threads per block, total {} CUDA threads\n", block_count, threads_per_block, core_count);
	fmt::print("Seeds per CUDA thread: {}\n", seeds_per_thread);
	fmt::print("Forcer context size: {} bytes\n",  context_size);

	void* alloc_base;
	gpuErrchk(cudaMalloc((void**)&alloc_base, context_size));

	unsigned rounds = seed_count / core_count;
	fmt::print("Rounds: {}\n", rounds);

	auto force_start = std::chrono::high_resolution_clock::now();
	auto start = force_start;
	for(unsigned i = 0; i < rounds; i++) {
		u32 seed_base = i * core_count;
		forcer_entrypoint<<<block_count, threads_per_block>>>(alloc_base, seed_base);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		auto eta = ((rounds - i) * duration) / 1000;
		auto eta_mins = eta / 60;
		auto eta_secs = eta % 60;
		if((i % 8) == 0) {
			const auto since_start = std::chrono::duration_cast<std::chrono::seconds>(end - force_start).count();
			const auto sps = (since_start == 0) ? 0 : ((i+1)*core_count) / since_start;
			fmt::print("Progress: round {}/{}, seeds: {:07x}x-{:07x}x [{}%], eta={}m:{}s, {} seeds/s\n", i, rounds, seed_base, (seed_base + core_count), 100.0 * seed_base / seed_count,
			  eta_mins, eta_secs, sps);
		}
		start = end;
	}

	cudaFree(alloc_base);
	return 0;
}
