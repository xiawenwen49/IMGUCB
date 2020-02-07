import os
from Oracle.generalGreedy import generalGreedy
from Oracle.degreeDiscount import degreeDiscountIC, degreeDiscountIC2, degreeDiscountIAC, degreeDiscountIAC2, degreeDiscountStar, degreeDiscountIAC3

# save_address = "./SimulationResults"

Flickr_data_dir = '/home/xiawenwen/datasets/Flickr'
NetHEPT_data_dir = '/home/xiawenwen/datasets/NetHEPT'
HEPPH_data_dir = '/home/xiawenwen/datasets/HEPPH'
DBLP_dir = '/home/xiawenwen/datasets/DBLP'

graph_address = 'Small_Final_SubG.G'
node_feature_address = 'Small_nodeFeatures.dic'
edge_feature_address = 'Small_edgeFeatures.dic'

linear_prob_address = 'Probability.dic'
nonlinear_prob_address = 'Probability_nonlinear.dic'


oracle = degreeDiscountIAC3
# oracle = degreeDiscountIAC2
# oracle = degreeDiscountIAC

alpha_1 = 0.1
# alpha_2 = 0.1
lambda_ = 0.4
# gamma = 0.1


sigma = 1
gamma = 0.1
var = 1
node_dimension = 20
edge_dimension = node_dimension*2
kernel_size = 2000
seed_size = 300
iterations = 500

# c = 1
c = 2
# c = 0.2

