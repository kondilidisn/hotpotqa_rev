import json

limit = 15000
i = 0

downsampled = []
with open('hotpot_train_v1.1.json') as json_data:
    d = json.load(json_data)
    # print(d)
    # print(len(d))
    for k in d:
    	downsampled.append(k)
    	i += 1
    	if i == limit:
    		break
    print(i)
    # save the downsampled json 
    with open('downsampled_training_set.json', 'w') as fp:
    	json.dump(downsampled, fp)
    	# print('1st level key :\n',k)
    	# for j in d[k]:
    	# 	print('2nd level key :',j)