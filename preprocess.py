def preprocess_w_lat_lons(netcdf_file_path):
    raw_counts = np.array(f.variables['raw_counts'])
            
    # Calcolo distanza tra il punto speculare e l'aereo
    ac_alt_2d = np.repeat(np.array(f.variables['ac_alt'])[:, np.newaxis], 20, axis=1)
    distance_2d = (ac_alt_2d - f.variables['sp_alt'][:]) / np.cos(np.deg2rad(f.variables['sp_inc_angle'][:]))

    # Seleziono gli indici dove sp_rx_gain_copol > 5, sp_rx_gain_xpol > 5 e ddm_snr > 0 e distanza tra punto speculare e antenna > 2000 e < 10000

    copol = f.variables['sp_rx_gain_copol'][:]
    xpol = f.variables['sp_rx_gain_xpol'][:]
    snr = f.variables['ddm_snr'][:]
    dist = distance_2d[:]
    specular_point_lat = f.variables['sp_lat'][:]
    specular_point_lon = f.variables['sp_lon'][:]
    

    keep_mask = (copol >= 5) & (xpol >= 5) & (snr > 0) & ((dist >= 2000) & (dist <= 10000)) & (~np.isnan(copol.data) & ~np.isnan(xpol.data) & ~np.isnan(snr.data) & ~np.isnan(dist.data) & ~np.isnan(specular_point_lat.data) & ~np.isnan(specular_point_lon.data))
    to_keep_indices = np.argwhere(keep_mask)
    
    filtered_raw_counts = [raw_counts[i, j] for i, j in to_keep_indices]
    output_array = np.full(raw_counts.shape, np.nan, dtype=np.float32)
    specular_point_lats = specular_point_lat[to_keep_indices[:, 0]]
    specular_point_lons = specular_point_lon[to_keep_indices[:, 0]]
    
    for idx, (i, j) in enumerate(to_keep_indices):
        output_array[i, j] = filtered_raw_counts[idx]
            
        raw_counts_filtered = output_array.copy()

    raw_counts_filtered = output_array.copy()
    del output_array

    ddm_data_dict = {
        'Raw_Counts': raw_counts_filtered.reshape(raw_counts_filtered.shape[0]*raw_counts_filtered.shape[1], raw_counts_filtered.shape[2], raw_counts_filtered.shape[3]),
    }

    keep_indices = np.where(
        np.all(~np.isnan(ddm_data_dict['Raw_Counts']), axis=(1, 2)) & (np.sum(ddm_data_dict['Raw_Counts'], axis=(1, 2)) > 0)
    )[0]
    fit_data = np.array([ddm_data_dict['Raw_Counts'][f].ravel() for f in keep_indices])

    specular_point_lats = specular_point_lat.ravel()[keep_indices]
    specular_point_lons = specular_point_lon.ravel()[keep_indices]
    
    surface_types = f.variables["sp_surface_type"][:]
    surface_types = np.nan_to_num(surface_types, nan=0)
    surface_types_unravelled = surface_types.ravel()
    label_data = [1 if surface_type in np.arange(1, 8) else 0 for surface_type in surface_types_unravelled]
    label_data = [label_data[lab] for lab in range(len(label_data)) if lab in keep_indices]

    assert np.array(fit_data).shape[0] == len(label_data) == np.array(specular_point_lats).shape[0] == np.array(specular_point_lons).shape[0], \
        f"Shape mismatch: fit_data {np.array(fit_data).shape[0]}, label_data {len(label_data)}, lats {np.array(specular_point_lats).shape[0]}, lons {np.array(specular_point_lons).shape[0]}"
    

    return fit_data, label_data, specular_point_lats, specular_point_lons