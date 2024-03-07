data = {
      "syn-COLG":[
        [0.1,23.30612244897959],
        [0.55, 20],
        [3, 2],
        [0.1, 0.35]
      ],
      "VAG":[
        [0.1, 0.35],
        [3, 2],
        [5, 1],
        [5, 0.05],
        [0.1, 0.05]
      ],
      "WPG-1":[
        [0.1,23.30612244897959],
        [0.55, 20],
        [3, 2],
        [100, 20],
        [100, 23.30612244897959]
      ],
      "WPG-2":[
        [100, 20],
        [3, 2],
        [5, 1],
        [100, 7]
      ],
      "ORG":[        
        [100, 7],
        [5, 1],
        [5, 0.05],
        [100, 0.05]
      ]
}

x_values = []
y_values = []

for key in data:
    for pair in data[key]:
        x_values.append(pair[0])
        y_values.append(pair[1])

x_min = min(x_values)
x_max = max(x_values)
y_min = min(y_values)
y_max = max(y_values)

# print(f"x range: {x_min} to {x_max}")
# print(f"y range: {y_min} to {y_max}")

print(x_min,',',x_max)
print(y_min,',',y_max)