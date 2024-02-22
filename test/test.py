from network import *

print('\n-------------------------------------------------------------------------------\n ')

xs = [
    [-3.0, 2.0 , 1.0],
    [3.2 , 1.0 , 3.0],
    [0.0 , 10.4, 7.2 ],
    [9.0 , 1.0 , -11.7 ]
]

labels = [
    1.0,
    -1.0,
    1.0,
    -1.0
]

l =  Value(0)

while(l.data<6):
  Model = MLP(3, [4, 4, 1], Value.tanh)
  l = basic_loss(Model.sample(xs), labels)
print(f'Original loss : {l}')

train(Model, xs, labels, basic_loss, 1000, 0.15, 1/100)

l = basic_loss(Model.sample(xs), labels)
print(Model.sample(xs))
print(l)


print('\n-------------------------------------------------------------------------------\n ')
