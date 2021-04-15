target_frame = 0xC3 * 3600 + 0x82 * 60 + 0xb8
frame_delay = 141

state = null
do_press = 0

movie.unsafe_rewind()

function on_set_rewind(new_state)
	state = new_state
end

function on_frame()
	m = memory.readbyte("WRAM", 0x1a43)
	s = memory.readbyte("WRAM", 0x1a44)
	f = memory.readbyte("WRAM", 0x1a45)
	global_frame = m * 3600 + s * 60 + f

	if (global_frame + frame_delay) == target_frame then
		do_press = 1
	end
end

function on_input(subframe)
	if do_press == 1 then
		input.set2(0,1,1,1)
		do_press = 0
	end
end

--  1 frame delay 
--  38,03
--  18,1F
--  19,36


--  27,03
--  28,1A
--  29,18

--  31,38	2996
--  34,11	3137 (+141)

--  2c,01   2641
--  2e,16   2782 (+141)

--  34,08   3128
--  36,1d   3269 (+141)

--  141 frame delay 