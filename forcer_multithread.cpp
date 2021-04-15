#include <list>
#include <mutex>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <fmt/format.h>
#include <thread>
#include <unistd.h>

typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

class FoolsRng {
    u8 m_state[4];
public:
    FoolsRng(u32 state) {
        reinitialize(state);
        // fmt::print("Init with {:x} ", m_state[3] | ((u32)m_state[2]<<8u) | ((u32)m_state[1]<<16u) | ((u32)m_state[0]<<24u));
    }
        
    u8 next() {
        m_state[0]++;
        m_state[1] = m_state[3] ^ m_state[0] ^ m_state[1];
        m_state[2] = m_state[1] + m_state[2];
        m_state[3] = m_state[3] + ((m_state[2]>>1) ^ m_state[1]);
        
        // fmt::print("new_state={:x} ", state());

        return m_state[3];
    }
    
    u32 state() const {
        return m_state[3] | ((u32)m_state[2]<<8u) | ((u32)m_state[1]<<16u) | ((u32)m_state[0]<<24u);
    }

    void reinitialize(u32 state) {
        m_state[3] = state & 0xffu;
        m_state[2] = (state >> 8u) & 0xffu;
        m_state[1] = (state >> 16u) & 0xffu;
        m_state[0] = (state >> 24u) & 0xffu;

        for(unsigned i = 0; i < 0x10; ++i)
            next();
    }
};

u8 gen_ctl(u32 seed, u16 x, u16 y) {
    u8 masked_x = x & 3,
       masked_y = y & 3;

    u32 mask = ((u32)(x & 0xFFFC) << 16u) | (y & 0xFFFC);
    FoolsRng rng {seed ^ mask};
    u8 b1 = rng.next() & 7;
    // fmt::print("b1={:x} ", b1);

    static const u8 dd0c_lookup[128] = {
        0x05,0x0B,0x06,0x00,0x0E,0x05,0x0B,0x03,0x09,0x0E,0x00,0x00,0x00,0x0C,0x00,0x00,0x00,0x0C,0x05,0x06,0x06,0x0D,0x0A,0x09,0x09,0x0B,0x06,0x00,0x00,0x05,0x0A,0x00,0x05,0x0B,0x06,0x00,0x0B,0x06,0x09,0x07,0x00,0x09,0x07,0x0A,0x00,0x05,0x0A,0x00,0x00,0x09,0x06,0x00,0x03,0x06,0x0C,0x05,0x00,0x09,0x0F,0x0A,0x00,0x05,0x0A,0x00,0x05,0x0A,0x00,0x00,0x0F,0x06,0x05,0x03,0x09,0x0F,0x0A,0x00,0x00,0x0C,0x00,0x00,0x05,0x0B,0x03,0x06,0x0E,0x00,0x00,0x0D,0x09,0x06,0x00,0x0C,0x00,0x0D,0x03,0x0A,0x00,0x09,0x06,0x00,0x06,0x05,0x0A,0x05,0x09,0x0F,0x06,0x0C,0x00,0x0C,0x09,0x0A,0x00,0x0C,0x00,0x00,0x03,0x0F,0x07,0x03,0x00,0x0D,0x0A,0x00,0x00,0x0C,0x00,0x00
    };

    auto offset = 4 * masked_y + masked_x;
    // fmt::print("offset={:x} ", offset);
    auto table_offset = offset + (b1 << 4u);
    // fmt::print("table_offset={:x} ", table_offset);
        
    u8 lookup_value = dd0c_lookup[table_offset];
    u8 b2 = rng.next() & 0x30;
    b2 |= lookup_value;
    // fmt::print("b2={:x}\n", b2);

    auto result = (((x&0xFF) | (y&0xFF)) & 0xFC) | ((x>>8u) | (y>>8u));
    if(result != 0)
        return b2;
    else
        return b2 & 0x0F;
}

struct PotentialSeeds {
	u32 seed;
	u8 x;
	u8 y;
};

struct Progress {
	unsigned done;
	unsigned count;
};

static thread_local u8 s_work_buffer[8 * 8];
static thread_local u8 s_decompress_buffer[0x30];
static thread_local u8 s_preloaded_map[64 * 9];

static std::vector<Progress> s_progress;
static std::mutex s_progress_lock;
static std::list<PotentialSeeds> s_potential_seeds;
static std::mutex s_potential_lock;

void map_decompress(FoolsRng& rng, u8 new_tile) {
    //  Copy tiles to not trample over things while modifying
    memcpy(&s_decompress_buffer[0], &s_work_buffer[8], 0x30);

    for(unsigned i = 8; i < 8*7; ++i) {
        if(s_decompress_buffer[i - 8] != new_tile)
            continue;
        if(((i&7) == 0) || ((i&7) == 7))
            continue;

        u8 v = rng.next();
        if(v & 1) {
            s_work_buffer[i-1] = new_tile;
        }
        if(v & 2) {
            s_work_buffer[i+1] = new_tile;
        }
        if(v & 4) {
            s_work_buffer[i-8] = new_tile;
        }
        if(v & 8) {
            s_work_buffer[i+8] = new_tile;
        }
    }
}

void map_place_tile_prob(FoolsRng& rng, u8 old, u8 new_tile, u8 threshold) {
    for(unsigned i = 0; i < 8*8; ++i) {
        if(s_work_buffer[i] != old)
            continue;
        u8 v = rng.next();
        if(v >= threshold)
            continue;

        s_work_buffer[i] = new_tile;
    }
}

//  Same as above, but skips lines 0 and 7 and tiles 0,7 on each line 
//  presumably to avoid softlocking
void map_place_tile_prob_safe(FoolsRng& rng, u8 old, u8 new_tile, u8 threshold) {
    for(unsigned i = 8; i < 8*7; ++i) {
        if((i&7) == 0 || (i&7) == 7)
            continue;
        if(s_work_buffer[i] != old)
            continue;
        u8 v = rng.next();
        if(v >= threshold)
            continue;

        s_work_buffer[i] = new_tile;
    }
}

void map_place_tile_at(u8 tile, u8 x, u8 y) {
    x = x & 0x0f;
    y = y & 0x0f;
    s_work_buffer[y * 8 + x] = tile;
    // fmt::print("place tile {:x} at {}x{}\n", tile, x,y);
}


void map_place_line_impl(u8 tile, u8 pos1, u8 pos2) {
    u8 x1 = (pos1 >> 4u) & 0xf,
       x2 = (pos2 >> 4u) & 0xf,
       y1 = pos1 & 0x0f,
       y2 = pos2 & 0x0f;

    u8 ystep = (y1 < y2) ? 0x01 : 0xFF,
       xstep = (x1 < x2) ? 0x01 : 0xFF;
    while((x1 != x2) || (y1 != y2)) {
        map_place_tile_at(tile, x1, y1);
        if(x1 != x2)
            x1 += xstep;
        map_place_tile_at(tile, x1, y1);
        if(y1 != y2)
            y1 += ystep;
        map_place_tile_at(tile, x1, y1);
    }
}

void map_place_line(FoolsRng& rng, u8 tile, u8 start, u8 end) {
    u8 x = rng.next() & 7;
    while(x == 0 || x == 7)
        x = rng.next() & 7;
    u8 y = rng.next() & 7;
    while(y == 0 || y == 7)
        y = rng.next() & 7;
 
    u8 point_pos = (x << 4u) | y;

    map_place_line_impl(tile, start, point_pos);
    map_place_line_impl(tile, point_pos, end);
}


void dump_current_map() {
    for(unsigned i = 0; i < 8*8; ++i) {
        fmt::print("{:02x},", s_work_buffer[i]); 
        if(i % 8 == 7)
            fmt::print("\n");
    }
}

void maybe_map_gen(FoolsRng& rng, u8 old, u8 new_tile, u8 threshold, u8 hfindpathflags, u8 hmultiplier, u8 hfindpathxprogress, u8 hmultiplybuffer) {
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
        if(s_work_buffer[i] != old) {
            i++; 
            c--; 
            continue;
        }

        u8 v = rng.next();
        if(v < threshold) {
            i++;
            c--; 
            continue;
        }

        i -= 0x8;
        if(hfindpathflags != 0) {
            if(s_work_buffer[i] != hfindpathflags) {
                i += 0x9; 
                c--;
                continue;
            }
        }

        i += 0x10;
        if(hmultiplier != 0) {
            if(s_work_buffer[i] != hmultiplier) {
                i -= 0x7;
                c--;
                continue;
            }
        }

        i -= 0x09;
        if(hfindpathxprogress != 0) {
            if(s_work_buffer[i] != hfindpathxprogress) {
                i += 2;
                c--;
                continue;
            }
        }

        i += 2;
        if(hmultiplybuffer != 0) {
            if(s_work_buffer[i] != hmultiplybuffer) {
                c--;
                continue;
            }
        }

        s_work_buffer[i-1] = new_tile;
        c--;
    }
}


void a901(FoolsRng& rng, u8 tile, u8 pos1, u8 pos2) {
    u8 x1 = (pos1 >> 4u) & 0xf,
       x2 = (pos2 >> 4u) & 0xf,
       y1 = pos1 & 0x0f,
       y2 = pos2 & 0x0f;

    u8 ystep = (y2 < y1) ? 0xff : 0x01,
       xstep = (x2 < x1) ? 0xff : 0x01;
    while((x1 != x2) || (y1 != y2)) {
        map_place_tile_at(tile, x1, y1);
        if(x1 != x2)
            x1 += xstep;
        map_place_tile_at(tile, x1, y1);
        if(y1 != y2)
            y1 += ystep;
        map_place_tile_at(tile, x1, y1);
    }
}

void gen_map_data(u32 seed, u16 x, u16 y) {
    //fmt::print("Generating map[{},{}] with seed={:08x}\n", x, y, seed);

    //  Generate control byte
    auto ctl = gen_ctl(seed, x,y);
    //fmt::print("mapctl={:x}\n", ctl);

    //  Fill work buffer with 0F
    memset(&s_work_buffer[0], 0x0F, 8 * 8);

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
    FoolsRng rng {seed ^ seed_mask};

    //fmt::print("seed_mask={:x} \n", seed_mask);
    //fmt::print("maprng first state={:x}\n", rng.state());

    //  World gen?
    if(ctl & 1)
        map_place_line(rng, 0xA, var1, 0x74);
    if(ctl & 2) 
        map_place_line(rng, 0xA, var1, 0x04);
    if(ctl & 4) 
        map_place_line(rng, 0xA, var1, 0x47);
    if(ctl & 8) 
        map_place_line(rng, 0xA, var1, 0x40);
    
    // fmt::print("after line place:\n");
    // dump_current_map();

    //  "Decompression"?
    map_decompress(rng, 0x0A);

    // fmt::print("after decompress:\n");
    // dump_current_map();


    //  Place exits 
    if(ctl & 8) {
        map_place_tile_at(0xa, 0x3, 0x0);
        map_place_tile_at(0xa, 0x4, 0x0); 
    }
    if(ctl & 4) {
        map_place_tile_at(0xa, 0x3, 0x7); 
        map_place_tile_at(0xa, 0x4, 0x7);
    }
    if(ctl & 2) {
        map_place_tile_at(0xa, 0x0, 0x3);
        map_place_tile_at(0xa, 0x0, 0x4);
    }
    if(ctl & 1) {
        map_place_tile_at(0xa, 0x7, 0x3);
        map_place_tile_at(0xa, 0x7, 0x4);
    }

    // fmt::print("after adding exits:\n");
    // dump_current_map();

    //  Biome specific generation 
    auto biome_ctl = (ctl >> 4u) & 0x3;
    //fmt::print("biome={:x}\n", biome_ctl);
    switch(biome_ctl) {
        case 0: {
            map_place_tile_prob(rng, 0xa, 0xb, 0x30);
            map_decompress(rng, 0x0B);
            maybe_map_gen(rng, 0x0f, 0x6c, 0x20, 0x0f, 0x0a, 0x0, 0x0);
            maybe_map_gen(rng, 0x0f, 0x6f, 0x20, 0x0a, 0x0f, 0x0, 0x0);
            maybe_map_gen(rng, 0x0f, 0x6e, 0x20, 0x0, 0x0, 0x0a, 0x0f);
            maybe_map_gen(rng, 0x0f, 0x6d, 0x20, 0x0, 0x0, 0x0f, 0x0a);
            map_place_tile_prob(rng, 0xa, 0x74, 0x30);
            map_place_tile_prob(rng, 0xa, 0x7a, 0x30);
            map_place_tile_prob_safe(rng, 0x6c, 0x33, 0x40);
            map_place_tile_prob_safe(rng, 0x6d, 0x32, 0x40);
            map_place_tile_prob_safe(rng, 0x6e, 0x60, 0x40);
            map_place_tile_prob_safe(rng, 0x6f, 0x34, 0x40);

            break;
        }
        case 1: {
            map_place_tile_prob(rng, 0xa, 0x7b, 0x40);
            map_place_tile_prob(rng, 0xa, 0x7a, 0x30);
            map_place_tile_prob(rng, 0xa, 0xb, 0xd0);
            map_place_tile_prob_safe(rng, 0x0a, 0x08, 0x20);
            break;
        }
        case 2: {
            //  Screw this
            assert(false);
            
            // maybe_map_gen(rng, 0xf, 0x13, 0xc0, 0x0, 0x0, 0x0, 0x0a);
            // maybe_map_gen(rng, 0xf, 0x13, 0xc0, 0x0, 0x0, 0x0a, 0x0);

            // maybe_map_gen(rng, 0x0a, 0x4e, 0x60, 0x0, 0x0, 0x0f, 0x0);
            // maybe_map_gen(rng, 0x0a, 0x4d, 0x60, 0x0, 0x0, 0x0, 0x0f);

            // maybe_map_gen(rng, 0x0a, 0x51, 0x60, 0x0f, 0x0, 0x0, 0x0);
            // maybe_map_gen(rng, 0x0a, 0x51, 0x60, 0x0, 0x0f, 0x0, 0x0);
            break;
        }
        case 3: {
            maybe_map_gen(rng, 0x0f, 0xec, 0x40, 0x0, 0x0, 0xa, 0x0);
            maybe_map_gen(rng, 0x0f, 0xec, 0x40, 0x0, 0x0, 0x0, 0xa);

            maybe_map_gen(rng, 0x0f, 0xec, 0x40, 0xa, 0x0, 0x0, 0x0);
            maybe_map_gen(rng, 0x0f, 0xec, 0x40, 0x0, 0xa, 0x0, 0x0);

            static const u8 add0_lookup[8] = {0xD9,0xDB,0xCA,0xCB,0x21,0x70,0xD1,0x06};

            map_place_tile_prob(rng, 0xec, 0xc9, 0x80);
            map_place_tile_prob(rng, 0x0a, 0x0b, 0x80);

            for(unsigned i = 0; i < 0x40; ++i) {
                if(s_work_buffer[i] != 0x0f) {
                    continue;
                }

                u8 b = rng.next();
                if(b >= 0x40)
                    continue;

                u8 val = rng.next() & 3u;
                s_work_buffer[i] = add0_lookup[val];
            }

            //  FIXME:  call a901
            if(ctl & 8) {
                a901(rng, 0x31, var1, 0x40);
            }
            if(ctl & 4) {
                a901(rng, 0x31, var1, 0x47);
            }
            if(ctl & 2) {
                a901(rng, 0x31, var1, 0x04);
            }
            if(ctl & 1) {
                a901(rng, 0x31, var1, 0x74);
            }

            map_place_tile_prob(rng, 0x31, 0x0b, 0x20);
            map_place_tile_prob_safe(rng, 0x31, 0x08, 0x10);

            break;
        }
        default: break;
    }

    //fmt::print("after biome:\n");
    //dump_current_map();
}

void fancy_mapview() {
    for(unsigned i = 0; i < 3*8; ++i) {
        for(unsigned j = 0; j < 3*8; ++j) {
            auto addr = i * 24 + j;
            fmt::print("{:02x}", s_preloaded_map[addr]);
            if((j % 8) == 7)
                fmt::print("|");
        }
        fmt::print("\n");
        if((i % 8) == 7) {
            for(unsigned z = 0; z < 3*8*2 + 3; z++) fmt::print("-");
            fmt::print("\n");
        }

    }
}

void generate_visible_map(u32 seed) {
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
            gen_map_data(seed, x, y+1);
            auto base = get_base(x,y);

            //  Copy lines such that the lines in nearby chunks are sequential in memory
            for(unsigned i = 0; i < 8; ++i) memcpy(&s_preloaded_map[base + i * 0x18], &s_work_buffer[i * 8], 8);
        }
    }
}

void forcer_entrypoint(unsigned tid, unsigned seed_begin, unsigned seed_count) {
	for(u32 i = seed_begin; i < seed_begin + seed_count; ++i) {
		//  Update progress
		if((i & 0x1fff) == 0) {
//			fmt::print("Thread {} - Progress: {} / {} [{}%]\n", tid, i, 0xfffffff, (double)i/0xfffffff);
			s_progress_lock.lock();
			s_progress[tid] = { .done = i - seed_begin, .count = seed_count};
			s_progress_lock.unlock();
		}

		auto seed = (i << 4u) | 0x01u;
		generate_visible_map(seed);

		const u8 sequence[25] = {
				0x0b,0x0b,0x0b,0x74,0x0a,
				0x0f,0x0b,0x0f,0x0a,0x0a,
				0x0f,0x0f,0x0a,0x0a,0x0b,
				0x0f,0x0b,0x0a,0x0a,0x0a,
				0x0b,0x0b,0x0a,0x0a,0x74
		};

		auto search_at = [&sequence](unsigned windowx, unsigned windowy) -> bool {
			if(windowx >= 24 || windowx + 5 >= 24 || windowy >= 24 || windowy + 5 >= 24)
				return false;

			unsigned c = 0;
			for(unsigned y = windowy; y < windowy + 5; ++y) {
				for(unsigned x = windowx; x < windowx + 5; ++x) {
					auto addr = y * 24 + x;
					if(s_preloaded_map[addr] != sequence[c])
						return false;
					c++;
					if(c == 25)
						return true;
				}
			}

			return false;
		};

		//  Search the possible window for the sequence
		for(unsigned x = 6; x <= 13; ++x) {
			for(unsigned y = 6; y <= 13; ++y) {
				bool res = search_at(x,y);
				if(res) {
					s_potential_lock.lock();
					s_potential_seeds.push_back({.seed = seed, .x = (u8)x, .y = (u8)y});
					fmt::print("SEED={:08x}/ Found pattern occurence at [{},{}]!\n", seed, x, y);
					s_potential_lock.unlock();
				}
			}
		}
	}
}

void look_for_cool_values_for_ace(u32 seed) {
    u8 ctl = gen_ctl(seed, 0x00B8, 0x0083);
    u8 biome_ctl = (ctl >> 4u) & 0x3;

    if(biome_ctl != 3) return;

    gen_map_data(seed, 0x00B8, 0x0083);

    unsigned off = 3 * 8 + 4;
    if(s_work_buffer[off] == 0x0f && s_work_buffer[off+1] == 0xdb)  {
        gen_map_data(seed, 0x00B8, 0x0082);
        bool can_move = false;
        for(unsigned i = 0; i < 64; ++i) 
            can_move |= (s_work_buffer[i] == 0x0A);

        if(can_move)
            fmt::print("Found pattern in map={{{:x},{:x}}}, seed={:08x}\n", 0x00B8, 0x0083, seed);
    }
}



int main() {
    const unsigned thread_count = 12;

    auto seed_count = 0xfffffff + 1;
    auto seeds_per_thread = seed_count / thread_count;
	fmt::print("Bruteforce using {} threads, {} seeds per thread\n", thread_count, seeds_per_thread);

	s_progress.resize(thread_count);

	std::list<std::thread> threads;
	unsigned temp = 0;
	for(unsigned i = 0; i < thread_count; ++i) {
		if(i == thread_count-1)
			seeds_per_thread = seed_count - temp;

		auto range_start = i * seeds_per_thread;
		threads.emplace_back(forcer_entrypoint, i, range_start, seeds_per_thread);
		fmt::print("Thread {}: {} seeds [{:08x} - {:08x}]\n", i, seeds_per_thread, range_start, range_start+seeds_per_thread);
		temp += seeds_per_thread;
	}

	while(true) {
		sleep(5);
		s_progress_lock.lock();
		for(unsigned i = 0; i < s_progress.size(); ++i) {
			fmt::print("Progress - Thread {}: {}/{} [{}%]\n", i, s_progress[i].done,s_progress[i].count, 100.0 * s_progress[i].done/s_progress[i].count);
		}
		s_progress_lock.unlock();
	}

    return 0;
}
