import diffusion as diff

# total time in Myr
t = 0.1

# generate random encounter parameters sufficient for a time period t
# N.B. only need to do this once!
diff.generate_rand_params(t)

# evolve the eccentricity for a total time period t
diff.diffuse(t)

# plot snapshot at 0 Myr, 0.5*t Myr and 0.99*t Myr
diff.plot_diffusion([0, 0.5*t, 0.99*t])
