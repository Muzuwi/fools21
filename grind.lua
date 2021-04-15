state = 0
movie.unsafe_rewind()

encounter = 0
mapx = 0
item21count = 0
slotcount = 0
 
print(input.controller_info(0,1))
current_frame_function = null

function on_set_rewind(new_state)
	state = new_state
end


function read_common()
	encounter = memory.readbyte("WRAM", 0x1ac9)
	mapx = memory.readbyte("WRAM", 0x1ab4)
	item21count = memory.readbyte("WRAM", 0x1b0e)

	for c=1,40 do
		count = memory.readbyte("WRAM", 0x1ae5 + c * 2)
		if (count == 0xFF) or (c == 40) then
			slotcount = c
			break
		end
	end

end


function on_paint()
	gui.textHV(15, 20, string.format("Encounter: %02x", encounter), 0xff0000)	
	gui.textHV(15, 40, string.format("Map: %02x", mapx), 0xff0000)	
	gui.textHV(15, 60, string.format("Slots: %d", slotcount), 0xff0000)
end

bpulse = -1

function wait_for_finish()
	bpulse = bpulse * -1
	if encounter == 0 then
		current_frame_function = wait_for_encounter
		movie.unsafe_rewind()
	end
end

frame = 0
dir = 1
frameskip = 0

function wait_for_encounter()
	if encounter == 0 then
		frame = frame + 1
		if frame == 5 then
			dir = dir * -1
			frame = 0
		end
		return
	end

	if encounter ~= 1 then
		movie.unsafe_rewind(state)
		frameskip = math.random(10, 80)
		return
	end

	current_frame_function = wait_for_finish
end

current_frame_function = wait_for_encounter
printed_close = 0

function run_frame()
	if frameskip ~= 0 then
		frameskip = frameskip - 1
		return
	end

	if item21count == 99 then
		if printed_close ~= 1 then
			print("Item 21 max count reached, closing")
			printed_close = 1
		end
		return
	end


	current_frame_function()
end

function on_frame()
	read_common()
	run_frame()
	gui.repaint()
end

function on_input(subframe)
	if dir == -1 then
		input.set2(0,1,5,1)
	end 
	if dir == 1 then
		input.set2(0,1,4,1)
	end
	if bpulse == 1 then
		input.set2(0,1,1,1)
	end
end
