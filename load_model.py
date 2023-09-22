model_json = 'V/model-3-press-geopot-cor_east.json'
model_h5 = 'V/model-3-press-geopot-cor_east.h5'

# mode 1: in: diff ; out: value
# mode 2: in: diff ; out: diff
# mode 3: in: diff and value for variable = target,
#         otherwise diff ; out: value
first_order_learn_mode = 3

# If the order of learning is 1, the test procedure can have two modes:
# mode 1: a single node is randomly picked and the model computes the vertical
#         distribution of a given variable of all eight neighbors (=directions).
#         The node is selected so that all eight compass directions exist
# mode 2: a single node is randomly selected and the model computes
#         the vertical distribution of a given variable at the furthest nodes away
#         from it. The latter are determined by the number of grid points away from
#         the selected node
first_order_test_mode = 2
number_grid_points_apart_test = 10
