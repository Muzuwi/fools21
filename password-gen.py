password = [0x0, 0x0, 0x0, 0xd2]

temp = 0xf9
temp2 = 0xd4
for i,byte in enumerate(password):
	byte = temp ^ temp2 ^ byte
	temp = temp2
	temp2 = byte
	password[i] = byte

checksum = ((((0x3b + password[0]) ^ password[1]) - password[2]) ^ password[3]) & 0xff

result = [password[0], password[1], password[2], password[3], checksum];
print('Result code: {:02x}{:02x}{:02x}{:02x}{:02x}'.format(password[0], password[1], password[2], password[3], checksum))
