import time
import Adafruit_ADXL345
import smbus
import Adafruit_BMP.BMP085 as BMP085

accel = Adafruit_ADXL345.ADXL345()

print('Printing X, Y, Z axis values, press Ctrl-C to quit...')


def clamp(v, minv, maxv):
    if v > maxv:
        return maxv
    elif v < minv:
        return minv
    return v


def get_signed_number(number):
    if number & (1 << 15):
        return number | ~65535
    else:
        return number & 65535


i2c_bus = smbus.SMBus(1)
i2c_address = 0x69
i2c_bus.write_byte_data(i2c_address, 0x20, 0x0F)
i2c_bus.write_byte_data(i2c_address, 0x23, 0x20)

bus = smbus.SMBus(1)
addrHMC = 0x1e


def read_word(address, adr):
    high = bus.read_byte_data(address, adr)
    low = bus.read_byte_data(address, adr + 1)
    val = (high << 8) + low
    return val


def read_word_2c(address, adr):
    val = read_word(address, adr)
    if val >= 0x8000:
        return -((65535 - val) + 1)
    else:
        return val


def main():
    bus.write_byte_data(addrHMC, 0, 0b01110000)
    bus.write_byte_data(addrHMC, 1, 0b00100000)
    bus.write_byte_data(addrHMC, 2, 0b00000000)

    while True:
        x_mag = read_word_2c(addrHMC, 3)
        y_mag = read_word_2c(addrHMC, 7)
        z_mag = read_word_2c(addrHMC, 5)
        x_mag = x_mag * 0.92
        y_mag = y_mag * 0.92
        z_mag = z_mag * 0.92

        x_raw, y_raw, z_raw = accel.read()
        x = x_raw / 8192
        y = y_raw / 8192
        z = z_raw / 8192
        x_acc = clamp(x, -2, 2)
        y_acc = clamp(y, -2, 2)
        z_acc = clamp(z, -2, 2)

        i2c_bus.write_byte(i2c_address, 0x28)
        x_l = i2c_bus.read_byte(i2c_address)
        i2c_bus.write_byte(i2c_address, 0x29)
        x_h = i2c_bus.read_byte(i2c_address)
        x = x_h << 8 | x_l
        i2c_bus.write_byte(i2c_address, 0x2A)
        y_l = i2c_bus.read_byte(i2c_address)
        i2c_bus.write_byte(i2c_address, 0x2B)
        y_h = i2c_bus.read_byte(i2c_address)
        y = y_h << 8 | y_l
        i2c_bus.write_byte(i2c_address, 0x2C)
        z_l = i2c_bus.read_byte(i2c_address)
        i2c_bus.write_byte(i2c_address, 0x2D)
        z_h = i2c_bus.read_byte(i2c_address)
        z = z_h << 8 | z_l
        x = get_signed_number(x)
        y = get_signed_number(y)
        z = get_signed_number(z)
        x_gyro = (x * 8.75) / 1000
        y_gyro = (y * 8.75) / 1000
        z_gyro = (z * 8.75) / 1000

        sensor = BMP085.BMP085()

        print('ACC: X={0}, Y={1}, Z={2}; GYRO: X={3}, Y={4}, Z={5}; MAG: X={6}, Y={7}, Z={8}; ALT: Temp = {9} *C, Pressure = {10} Pa, Altitude = {11} m, Sealevel Pressure = {12} Pa'.format(
                x_acc, y_acc, z_acc,
                x_gyro, y_gyro, z_gyro,
                x_mag, y_mag, z_mag, sensor.read_temperature(),
                sensor.read_pressure(), sensor.read_altitude(), sensor.read_sealevel_pressure()))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
