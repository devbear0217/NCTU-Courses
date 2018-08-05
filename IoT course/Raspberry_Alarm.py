import time
import Adafruit_ADXL345
import smbus
import math
import random
import Adafruit_BMP.BMP085 as BMP085
import RPi.GPIO as GPIO

accel = Adafruit_ADXL345.ADXL345()
a = 0


def clamp(v, min_v, max_v):
    if v > max_v:
        return max_v
    elif v < min_v:
        return min_v
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


def mag():
    x_mag = read_word_2c(addrHMC, 3)
    y_mag = read_word_2c(addrHMC, 7)
    z_mag = read_word_2c(addrHMC, 5)
    x_mag = x_mag * 0.92
    y_mag = y_mag * 0.92
    z_mag = z_mag * 0.92
    return x_mag, y_mag, z_mag


def acc():
    x_raw, y_raw, z_raw = accel.read()
    x = x_raw / 8192
    y = y_raw / 8192
    z = z_raw / 8192
    x_acc = clamp(x, -2, 2)
    y_acc = clamp(y, -2, 2)
    z_acc = clamp(z, -2, 2)
    return x_acc, y_acc, z_acc


def gyro():
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
    return x_gyro, y_gyro, z_gyro


def shaking(x_acc_start, y_acc_start, z_acc_start):
    x_acc, y_acc, z_acc = acc()
    if (x_acc > x_acc_start + 0.01 or x_acc < x_acc_start - 0.01) or (
            y_acc > y_acc_start + 0.01 or y_acc < y_acc_start - 0.01) or (
            z_acc > z_acc_start + 0.01 or z_acc < z_acc_start - 0.01):
        return 'succesfully'
    else:
        return 'unsuccesfully'


def heading(x_mag, y_mag):
    declination = -4.47
    pi = 3.14159265359

    heading = math.atan2(y_mag, x_mag) + declination

    if heading > 2 * pi:
        heading = heading - 2 * pi

    if heading < 0:
        heading = heading + 2 * pi

    heading_angle = int(heading * 180 / pi)

    return heading_angle


def dist(a, b):
    return math.sqrt((a * a) + (b * b))


def rotation(x_acc_start, y_acc_start, z_acc_start):
    m_pi = 3.14159265358979323846
    rad_to_deg = 57.29578
    acc_xangle = (math.atan2(y_acc_start, z_acc_start) + m_pi) * rad_to_deg
    acc_yangle = (math.atan2(z_acc_start, x_acc_start) + m_pi) * rad_to_deg
    return acc_xangle, acc_yangle


def main():
    bus.write_byte_data(addrHMC, 0, 0b01110000)
    bus.write_byte_data(addrHMC, 1, 0b00100000)
    bus.write_byte_data(addrHMC, 2, 0b00000000)

    while True:

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(18, GPIO.OUT)
        GPIO.setup(27, GPIO.OUT)

        start = time.time()
        tasks = ['rotation',
                 'shaking',
                 'heading',
                 'altitude']
        task = random.choice(tasks)
        while time.time() - start < 3:
            GPIO.output(18, GPIO.HIGH)
            GPIO.output(27, GPIO.LOW)

        while time.time() - start >= 3 and time.time() - start < 5:
            GPIO.output(18, GPIO.LOW)
            GPIO.output(27, GPIO.HIGH)
            if task == 'shaking':
                print('Please shake your device to turn alarm off')
                time.sleep(1)
                result = shaking(x_acc_start,
                                 y_acc_start,
                                 z_acc_start)
            elif task == 'heading':
                print('Please head your device to the north(90 degrees) to turn alarm off')
                time.sleep(1)
                x_mag, y_mag, z_mag = mag()
                result = heading(x_mag,
                                 y_mag,
                                 z_mag)
            elif task == 'altitude':
                sensor = BMP085.BMP085()
                start_alt = sensor.read_altitude()
                print('Please rise your device to turn alarm off')
                time.sleep(1)
                sensor = BMP085.BMP085()
                result = sensor.read_altitude()
            elif task == 'rotation':
                x_acc_start, y_acc_start, z_acc_start = acc()
                start_rot = rotation(x_acc_start,
                                     y_acc_start,
                                     z_acc_start)
                start_rot = [int(i) for i in start_rot]
                print('Please rotate your device to turn alarm off')
                time.sleep(1)
                x_acc_start, y_acc_start, z_acc_start = acc()
                result = rotation(x_acc_start,
                                  y_acc_start,
                                  z_acc_start)
                result = [int(i) for i in result]

        while time.time() - start >= 5:
            if (task == 'heading' and result not in range(85, 95)) or (
                    task == 'shaking' and result == 'unsuccesfully') or (
                    task == 'altitude' and result <= start_alt) or (task == 'rotation' and result == start_rot):
                print("You have failed the task")
                while True:
                    GPIO.output(18, GPIO.LOW)
                    GPIO.output(27, GPIO.HIGH)
            time.sleep(1)
            break


if __name__ == "__main__":
    main()
