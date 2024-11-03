from Indivisuals import Individual
from Products import Product

product1 = Product("Refrigerator", 0.751, 999.9)
print(product1.name, product1.space, product1.price)
product2 = Product("cell phone", 0.0000899, 2199.12)
print(product2.name, product2.space, product2.price)

products_list = [Product('Refrigerator A', 0.751, 999.90), Product('Cell phone', 0.00000899, 2199.12),
                 Product('TV 55', 0.400, 4346.99), Product("TV 50' ", 0.290, 3999.90),
                 Product("TV 42' ", 0.200, 2999.00), Product("Notebook A", 0.00350, 2499.90),
                 Product("Ventilator", 0.496, 199.90), Product("Microwave A", 0.0424, 308.66),
                 Product("Microwave B", 0.0544, 429.90), Product("Microwave C", 0.0319, 299.29),
                 Product("Refrigerator B", 0.635, 849.00), Product("Refrigerator C", 0.870, 1199.89),
                 Product("Notebook B", 0.498, 1999.90), Product("Notebook C", 0.527, 3999.00)]

print(products_list)
#
# for product in products_list:
#     print(product.name, product.space, product.price)

spaces = []
prices = []
names = []
for product in products_list:
    spaces.append(product.space)
    prices.append(product.price)
    names.append(product.name)
limit = 3

print(spaces)
print(prices)
print(names)
print(names[5], prices[5], spaces[5])

individual1 = Individual(spaces, prices, limit)
# print('Spaces: ', individual1.spaces)
# print('Prices: ', individual1.prices)
# print('Chromosome: ', individual1.chromosome)
for i in range(len(products_list)):
    # print(individual1.chromosome[i])
    if individual1.chromosome[i] == '1':
        print('Name: ', products_list[i].name)
individual1.fitness()
print('Score: ', individual1.score_evaluation)
print('Used space: ', individual1.used_space)
print('Chromosome: ', individual1.chromosome)
