import matplotlib.pyplot as plt
import json


losses = json.load(open('./results/losses.json', 'r'))
train_acc = json.load(open('./results/train_acc.json', 'r'))
test_acc = json.load(open('./results/test_acc.json', 'r'))
print(train_acc[-1])
plt.figure(figsize=(25, 10))
axes1 = plt.subplot(121)
plt.plot(losses, color='#57A9D1', label='loss', linewidth=3)
plt.xlabel('iter', fontsize=14)
plt.ylabel('loss', fontsize=14)
[axes1.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right']]
axes2 = plt.subplot(122)
plt.plot(train_acc, color='#57A9D1', label='Training Accuracy', linewidth=3)
plt.plot(test_acc, color='#FF6347', label='Testing Accuracy', linewidth=3)
plt.xlabel('iter', fontsize=14)
plt.ylabel('acc', fontsize=14)
[axes2.spines[loc_axis].set_visible(False) for loc_axis in ['top', 'right']]
plt.legend()
plt.show()