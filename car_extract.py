#coding=utf-8

import math

def read_files():
	brand_names = {'UNK':0}
	with open("/home/zhouyang/data/car/secondinfo.brand", 'r') as f:
		id = 1;
		for row in f:
			row = row.strip()
			assert row not in brand_names
			brand_names[row] = id;
			id = id + 1

	samples = []
	with open("/home/zhouyang/data/car/secondinfo.txt", 'r') as f:
		start = True
		line_num = 1
		for row in f:
			if start:
				start = False
				continue
			##print("%d:\t%s" % (line_num, row))
			line_num = line_num + 1
			(city, brand, car_title, offer_price, mileage, newcar_price, car_age, result) = \
					row.split()
			curid = 0
			for name, id in brand_names.items():
				if brand.find(name) >= 0:
					curid = id
					break
			jinkou = 1 if car_title.find('进口') >= 0 else 0
			no_jinkou = 1 - jinkou
			shoudong = 1 if car_title.find('手动') >= 0 else 0
			no_shoudong = 1 - shoudong

			mileage = float(mileage)
			mileage = 1.0 if mileage >= 60.0 else mileage / 60.0
			mileages = [math.sqrt(mileage), mileage, mileage * mileage]

			newcar_price = float(newcar_price)
			newcar_price = 1.0 if newcar_price >= 100.0 else newcar_price / 100.0
			newcar_prices = [math.sqrt(newcar_price), newcar_price, newcar_price * newcar_price]

			car_age = float(car_age)
			car_age = 1.0 if car_age >= 240.0 else car_age / 240.0
			car_ages = [math.sqrt(car_age), car_age, car_age * car_age]

			residual_rate = float(result)

			samples.append([curid, jinkou, no_jinkou, shoudong, no_shoudong] + mileages + newcar_prices + car_ages + [residual_rate])

	return brand_names, samples


