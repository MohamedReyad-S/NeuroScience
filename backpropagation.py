import numpy as np

# تعريف دالة التنشيط سيجمويد ومشتقتها
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# المدخلات
x1, x2 = 0.05, 0.10
y1, y2 = 0.01, 0.99

# الأوزان الأولية بين المدخلات والطبقة المخفية
w11, w12 = 0.15, 0.20
w21, w22 = 0.25, 0.30

# الأوزان الأولية بين الطبقة المخفية والطبقة النهائية
w31, w32 = 0.40, 0.45
w41, w42 = 0.50, 0.55

# الانحيازات
b1, b2 = 0.35, 0.35
b3, b4 = 0.60, 0.60

# معدل التعلم
learning_rate = 0.5

# ======= الانتشار الأمامي (Forward Propagation) =======
net_h1 = (x1 * w11) + (x2 * w21) + b1
net_h2 = (x1 * w12) + (x2 * w22) + b2

h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)

net_o1 = (h1 * w31) + (h2 * w41) + b3
net_o2 = (h1 * w32) + (h2 * w42) + b4

o1 = sigmoid(net_o1)
o2 = sigmoid(net_o2)

# ======= حساب الخطأ الكلي =======
E1 = 0.5 * (y1 - o1) ** 2
E2 = 0.5 * (y2 - o2) ** 2
E_total = E1 + E2

# ======= حساب التدرج العكسي (Backpropagation) =======
# التدرج للأوزان بين الطبقة المخفية والإخراج
dE_total_do1 = o1 - y1
dE_total_do2 = o2 - y2

do1_dneto1 = sigmoid_derivative(o1)
do2_dneto2 = sigmoid_derivative(o2)

dneto1_dw31 = h1
dneto1_dw32 = h2
dneto2_dw41 = h1
dneto2_dw42 = h2

dE_total_dw31 = dE_total_do1 * do1_dneto1 * dneto1_dw31
dE_total_dw32 = dE_total_do1 * do1_dneto1 * dneto1_dw32
dE_total_dw41 = dE_total_do2 * do2_dneto2 * dneto2_dw41
dE_total_dw42 = dE_total_do2 * do2_dneto2 * dneto2_dw42

# ======= طباعة التدرجات المحسوبة =======
print("\n🔹 التدرجات المحسوبة قبل تحديث الأوزان:")
print(f"∂E/∂w31: {dE_total_dw31:.6f}")
print(f"∂E/∂w32: {dE_total_dw32:.6f}")
print(f"∂E/∂w41: {dE_total_dw41:.6f}")
print(f"∂E/∂w42: {dE_total_dw42:.6f}")

# ======= تحديث الأوزان =======
w31 -= learning_rate * dE_total_dw31
w32 -= learning_rate * dE_total_dw32
w41 -= learning_rate * dE_total_dw41
w42 -= learning_rate * dE_total_dw42

# ======= طباعة القيم النهائية =======
print("\n🔹 القيم النهائية بعد تحديث الأوزان:")
print(f"o1: {o1:.4f}, o2: {o2:.4f}")
print(f"إجمالي الخطأ: {E_total:.6f}")

print("\n🔹 الأوزان الجديدة بين الطبقة المخفية والمخرجات:")
print(f"w31: {w31:.4f}, w32: {w32:.4f}, w41: {w41:.4f}, w42: {w42:.4f}")
