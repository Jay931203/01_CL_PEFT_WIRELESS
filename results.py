import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. NMSE 데이터 정의
nmse_data = {
    'city_6_miami':      [0.0349, 0.0278, 0.0748, 0.0196, 0.0381, 0.0082, 0.0422],
    'city_7_sandiego':   [0.0343, 0.0260, 0.0624, 0.0198, 0.0393, 0.0099, 0.0422],
    'city_11_santaclara':[0.0367, 0.0311, 0.0374, 0.0153, 0.0453, 0.0104, 0.0513],
    'city_12_fortworth': [0.0420, 0.0391, 0.0415, 0.0117, 0.0528, 0.0160, 0.0609],
    'city_15_indianapolis':[0.0287, 0.0230, 0.0339, 0.0118, 0.0269, 0.0101, 0.0393],
    'city_18_denver':    [0.0340, 0.0278, 0.0417, 0.0123, 0.0337, 0.0110, 0.0454],
    'city_19_oklahoma':  [0.0265, 0.0221, 0.0519, 0.0148, 0.0147, 0.0102, 0.0199]
}

# 2. 테스트 도시 순서 정의
test_cities = ['city_6_miami', 'city_7_sandiego', 'city_11_santaclara',
               'city_12_fortworth', 'city_15_indianapolis', 'city_18_denver', 'city_19_oklahoma']

# 3. DataFrame으로 변환
df_nmse = pd.DataFrame(nmse_data, index=test_cities)

# 4. Long format으로 변환
df_long = df_nmse.reset_index().melt(id_vars='index', var_name='Main City', value_name='NMSE')
df_long = df_long.rename(columns={'index': 'Test City'})

# 5. 도시 번호 매핑
city_order = {
    'city_6_miami': 6,
    'city_7_sandiego': 7,
    'city_11_santaclara': 11,
    'city_12_fortworth': 12,
    'city_15_indianapolis': 15,
    'city_18_denver': 18,
    'city_19_oklahoma': 19
}
df_long['Test City Number'] = df_long['Test City'].map(city_order)
df_long['Main City Number'] = df_long['Main City'].map(city_order)

# 6. Main City가 Test City 이상인 경우만 필터링 (continual learning처럼)
df_long_sorted = df_long.sort_values(by=['Test City Number', 'Main City Number'])
cl_fixed = df_long_sorted[df_long_sorted['Main City Number'] >= df_long_sorted['Test City Number']]

# 7. 그래프 그리기
plt.figure(figsize=(10, 6))

for test_city in cl_fixed['Test City'].unique():
    subset = cl_fixed[cl_fixed['Test City'] == test_city]
    label = f"{city_order[test_city]}_{test_city.split('_')[2]}"
    plt.plot(
        subset['Main City Number'],
        subset['NMSE'],
        marker='o',
        label=label
    )

plt.xticks(sorted(city_order.values()))
plt.xlabel("Main City Number (Trained On)")
plt.ylabel("NMSE")
plt.title("NMSE vs Trained City (Sequential Testing per Test City)")
plt.legend(title="Test City (Evaluated On)", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()