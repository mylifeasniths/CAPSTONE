import matplotlib.pyplot as plt

# this file generates graph for accuracy comparision between cnn and svm

plt.style.use('ggplot')
models = ['SVM', 'CNN']
accuracy = [97.920, 99.280]

x_pos = [i for i, _ in enumerate(models)]

plt.bar(x_pos, accuracy, color='orange',width=[0.5,0.5])
plt.xlabel("Modal type")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.xticks(x_pos, models)
plt.show()
