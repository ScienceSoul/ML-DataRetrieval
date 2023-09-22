from utils import *

eta = get_eta(num_vert_layers, path_to_eta)
plot_vertical_profile('U', time_stamps, int(sys.argv[1]), '.csv', eta, '(m s-1)', 661, 381, 210, False, y_log=True)
plot_vertical_profile('U', time_stamps, int(sys.argv[1]), '.csv', eta, '(m s-1)', 661, 382, 210, False, y_log=True)
plot_vertical_profile('U', time_stamps, int(sys.argv[1]), '_interpol.csv', eta, '(m s-1)', 660, 381, 210, False, y_log=True)

plot_vertical_profile('V', time_stamps, int(sys.argv[1]), '_interpol.csv', eta, '(m s-1)', 660, 381, 210, False, y_log=True)
plot_vertical_profile('V', time_stamps, int(sys.argv[1]), '_interpol.csv', eta, '(m s-1)', 660, 381, 211, False, 
                     y_log=True)
plot_vertical_profile('V', time_stamps, int(sys.argv[1]), '_interpol.csv', eta, '(m s-1)', 660, 381, 210, False, y_log=True)

plot_vertical_profile('W', time_stamps, int(sys.argv[1]), '.csv', eta, '(m s-1)', 660, 381, 210, False, y_log=True)

plot_vertical_profile('REL_VERT_VORT', time_stamps, int(sys.argv[1]), '.csv', eta, '(ks-1)', 660, 381, 210, False, y_log=True)

display_profile_abs_velocity(time_stamps, int(sys.argv[1]), '_interpol.csv', eta, '(m s-1)', 660, 381, 210, False, y_log=True)

plot_vertical_profile('PERT_T', time_stamps, int(sys.argv[1]), '.csv', eta, '(Pa)', 660, 381, 210, True, y_log=True)
