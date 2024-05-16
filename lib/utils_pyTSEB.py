'''
Utils functions to run TSEB
'''

import numpy as np
import tseb


#%%

def _run_tseb(indict):
    out_dict = {}
    sw_in_dir = indict[PAR_DIR] + indict[NIR_DIR]
    sw_in_dif = indict[PAR_DIF] + indict[NIR_DIF]
    fvis = (indict[PAR_DIR] + indict[PAR_DIF]) / (sw_in_dir + sw_in_dif)
    fvis = np.nan_to_num(fvis, 0)


     # Create output variables
    [out_dict[LE_C], out_dict[LE_0], out_dict[LE_C_0], out_dict[G],
     out_dict[G_ST], le_s, h_c, h_s, sn_c, sn_s, ln_c, ln_s] = map(
        np.zeros_like, 12 * [indict[LST]])
    out_dict[FLAG] = np.full_like(out_dict[LE_C], 255, dtype=int)

    soil = np.logical_and(np.logical_or(indict[LAI] == 0,
                                        indict[F_C] == 0),
                          indict[IGBP] != 17)

    log.info("Running OSEB for bare soil pixels")
    sn_s[soil] = (indict[PAR_DIR][soil] + indict[PAR_DIF][soil]) * \
                 (1 - indict[RHO_SOIL_VIS][soil]) + \
                 (indict[NIR_DIR][soil] + indict[NIR_DIF][soil]) * \
                 (1 - indict[RHO_SOIL_NIR][soil])

    del indict[NIR_DIR], indict[NIR_DIF]

    out_dict[FLAG][soil], ln_s[soil], le_s[soil], h_s[soil], out_dict[G][soil], *_ = tseb.OSEB(
        indict[LAI][soil], indict[TA][soil], indict[WS][soil],
        indict[EA][soil], indict[PA][soil], sn_s[soil], indict[LDN][soil],
        EMISS_S, Z_0_SOIL, 0, Z_U, Z_T)

    out_dict[LE_C][soil] = 0.0
    out_dict[LE_C_0][soil] = 0.0
    out_dict[LE_0][soil] = 0.0

    veg = indict[LAI] > 0
    log.info("Calculate clumping index for clumped canopies")
    clumped = np.logical_and.reduce((veg, indict[F_C] > 0, indict[F_C] < 1))
    # local LAI
    local_lai = indict[LAI][clumped] / indict[F_C][clumped]
    omega_0 = tseb.CI.calc_omega0_Kustas(local_lai,
                                         indict[F_C][clumped],
                                         x_LAD=indict[X_LAD][clumped],
                                         isLAIeff=True)
    omega = tseb.CI.calc_omega_Kustas(omega_0,
                                      indict[SZA][clumped],
                                      w_C=indict[WC_HC][clumped])
    lai_eff = np.copy(indict[LAI])
    lai_eff[clumped] = local_lai * omega
    del local_lai, omega, omega_0

    par_in = (indict[PAR_DIR] + indict[PAR_DIF]) / lf.MUEINSTEIN_2_WATT
    apar = (1. - indict[RHO_LEAF_VIS]) * par_in
    del par_in
    log.info("Get maximum stomata conductance")
    vpd = np.maximum(tseb.met.calc_vapor_pressure(indict[TA]) - indict[EA], 0)
    c_vcx, c_jx, c_rd = _photosynthesis_params(indict[IGBP])
    out_dict[G_ST_0] = lf.gpp_leaf_no_gs(indict[TA],
                                         vpd,
                                         apar,
                                         theta=PHOTO_DATA["THETA"],
                                         alpha=PHOTO_DATA["ALPHA"],
                                         c_kc=PHOTO_DATA["K_c"],
                                         c_ko=PHOTO_DATA["K_o"],
                                         c_tes=PHOTO_DATA["TES"],
                                         c_rd=c_rd,
                                         c_vcx=c_vcx,
                                         c_jx=c_jx,
                                         g0p=G_OP,
                                         a_1=PHOTO_DATA["G_S"]["a"],
                                         d_0=PHOTO_DATA["G_S"]["D_0"])[2]
    del vpd, apar, c_rd, c_vcx, c_jx
    # Convert conductance to CO2 to H2O
    out_dict[G_ST_0] = np.maximum(out_dict[G_ST_0], G_OP)

    # convert conductance to stomata resistance s m-1
    rst_min = 1. / (tseb.res.molm2s1_2_ms1(indict[TA],
                                           indict[PA]) *\
                    out_dict[G_ST_0] * lf.GV_GC_RATIO)
    rst_min[rst_min < 0] = np.nan
    log.info('Running Cambpbell radiation transfer')
    sn_c[veg], sn_s[veg] = tseb.rad.calc_Sn_Campbell(indict[LAI][veg],
                                                     indict[SZA][veg],
                                                     sw_in_dir[veg],
                                                     sw_in_dif[veg],
                                                     fvis[veg],
                                                     1 - fvis[veg],
                                                     indict[RHO_LEAF_VIS][veg],
                                                     indict[TAU_LEAF_VIS][veg],
                                                     indict[RHO_LEAF_NIR][veg],
                                                     indict[TAU_LEAF_NIR][veg],
                                                     indict[RHO_SOIL_VIS][veg],
                                                     indict[RHO_SOIL_NIR][veg],
                                                     x_LAD=indict[X_LAD][veg],
                                                     LAI_eff=lai_eff[veg])
    # Net radiation (Rn) for vegetation and soil
    sn_c[~np.isfinite(sn_c)] = 0
    sn_s[~np.isfinite(sn_s)] = 0
    del sw_in_dir, sw_in_dif, fvis, indict[SZA], lai_eff
    for var in SPECTRAL_VARS:
        del indict[var]

    log.info("Compute Aerodynamic roughness and zero-plane displacement height")
    z_0m, d_0 = tseb.res.calc_roughness(indict[LAI][veg],
                                        indict[H_C][veg],
                                        indict[WC_HC][veg],
                                        landcover=indict[IGBP][veg],
                                        f_c=indict[F_C][veg])
    del indict[IGBP]
    d_0[d_0 < 0] = 0
    z_0m[z_0m < Z_0_SOIL] = Z_0_SOIL
    alpha_pt = alpha_pt_komatsu(indict[H_C])
    alpha_pt[indict[H_C] < 10] = DEFAULT_ALPHA_PT

    log.info('Running TSEB')
    # TSEB model - left side OUTPUT, right side INPUT
    [out_dict[FLAG][veg], t_s, t_c, t_0, ln_s[veg], ln_c[veg],
     out_dict[LE_C][veg], h_c[veg], le_s[veg], h_s[veg], out_dict[G][veg],
     r_s, r_x, r_a, u_friction, l_mo, counter] = \
        tseb.TSEB_PT(indict[LST][veg],
                     indict[VZA][veg],
                     indict[TA][veg],
                     indict[WS][veg],
                     indict[EA][veg],
                     indict[PA][veg],
                     sn_c[veg],
                     sn_s[veg],
                     indict[LDN][veg],
                     indict[LAI][veg],
                     indict[H_C][veg],
                     EMISS_V,
                     EMISS_S,
                     z_0m,
                     d_0,
                     Z_T,
                     Z_U,
                     x_LAD=indict[X_LAD][veg],
                     f_c=indict[F_C][veg],
                     f_g=indict[F_G][veg],
                     w_C=indict[WC_HC][veg],
                     leaf_width=indict[LEAF_SIZE][veg],
                     z0_soil=Z_0_SOIL,
                     alpha_PT=alpha_pt[veg])
    for var in [LST, VZA]:
        del indict[var]
    del t_s, t_0, r_s, u_friction, counter, alpha_pt
    # Remove all values with returning arithmeting errors, recorded in flag=255
    no_data = out_dict[FLAG] == 255
    ln_s[no_data] = np.nan
    ln_c[no_data] = np.nan
    out_dict[RN] = sn_c + sn_s + ln_c + ln_s
    del ln_c, ln_s
    out_dict[LE_C][no_data] = np.nan
    le_s[no_data] = np.nan
    out_dict[LE] = out_dict[LE_C] + le_s
    del le_s
    h_c[no_data] = np.nan
    h_s[no_data] = np.nan
    out_dict[H] = h_c + h_s
    del h_c, h_s
    out_dict[G][no_data] = np.nan

    log.info('Running Shuttelworth & Wallace Potential fluxes')
    [_, _, _, _, _, _, out_dict[LE_0][veg], _, out_dict[LE_C_0][veg],
     _, _, _, _, _, _, _, _, _, _] = \
        pet.shuttleworth_wallace(indict[TA][veg],
                                 indict[WS][veg],
                                 indict[EA][veg],
                                 indict[PA][veg],
                                 sn_c[veg],
                                 sn_s[veg],
                                 indict[LDN][veg],
                                 indict[LAI][veg],
                                 indict[H_C][veg],
                                 EMISS_V,
                                 EMISS_S,
                                 z_0m,
                                 d_0,
                                 Z_T,
                                 Z_U,
                                 x_LAD=indict[X_LAD][veg],
                                 f_c=indict[F_C][veg],
                                 f_g=indict[F_G][veg],
                                 w_C=indict[WC_HC][veg],
                                 leaf_width=indict[LEAF_SIZE][veg],
                                 z0_soil=Z_0_SOIL,
                                 Rst_min=rst_min[veg],
                                 R_ss=500,
                                 const_L=l_mo,
                                 verbose=True)
    del _, rst_min, l_mo, sn_c, sn_s,
    for var in [WS, LDN, H_C, X_LAD, F_C, WC_HC, LEAF_SIZE]:
        del indict[var]

    values = np.logical_and.reduce((veg,
                                    np.isfinite(out_dict[LE_C]),
                                    np.isfinite(out_dict[LE])))

    out_dict[G_ST][values] = tseb.res.calc_stomatal_conductance_TSEB(
        out_dict[LE_C][values], out_dict[LE][values], r_a[values], r_x[values],
        indict[EA][values], indict[TA][values], t_c[values],
        indict[LAI][values],
        p=indict[PA][values],
        leaf_type=1,
        f_g=indict[F_G][values],
        f_dry=1,
        max_gs=1)

    out_dict[G_ST] = np.maximum(out_dict[G_ST] / lf.GV_GC_RATIO, G_OP)
    out_dict[G_ST] = np.nan_to_num(out_dict[G_ST], nan=G_OP, neginf=G_OP)
    return out_dict


def daily_fluxes(date_obj,
                 tseb_dir,
                 meteo_hourly_folder,
                 meteo_daily_folder,
                 is_high_resolution=False):

    date_str = date_obj.strftime("%Y%m%d")
    sdn_daily_file = meteo_daily_folder / f"{date_str}_LEVEL2_ECMWF_SDNday.tif"
    ta_daily_file = meteo_daily_folder / f"{date_str}_LEVEL2_ECMWF_TAday.tif"
    tseb_scenes = list(tseb_dir.glob(f"{date_str}T*"))
    template_file = list(list(tseb_scenes)[0].glob("*_FLAG.tif"))[0]
    proj, gt, xsize, ysize, *_ = gu.raster_info(template_file)
    le_ratio = []
    le_c_ratio = []
    le_0_ratio = []
    le_c_0_ratio = []
    for tseb_scene in tseb_scenes:
        datetime = tseb_scene.stem
        meteo_dir = meteo_hourly_folder / datetime
        log.info(f"Get instantaneous irradiance for {tseb_scene}")
        sdn = np.zeros([ysize, xsize])
        for var in ["PAR-DIR", "PAR-DIF", "NIR-DIR" , "NIR-DIF"]:
            meteo_file = meteo_dir / f'{datetime}_LEVEL2_ECMWF_{var}.tif'
            if is_high_resolution:
                fid = gu.resample_with_gdalwarp(meteo_file,
                                                template_file,
                                                "bilinear",
                                                "MEM")
                values = fid.GetRasterBand(1).ReadAsArray()
                del fid
            else:
                values = gu.get_raster_data(meteo_file)
            sdn = sdn + np.maximum(np.nan_to_num(values, 0), 0)

        flag_file = list(tseb_scene.glob("*_FLAG.tif"))[0]
        date, level, satellite, *_ = flag_file.stem.split("_")
        flag = gu.get_raster_data(flag_file)
        valid = np.isin(flag, VALID_FLAGS)

        in_file = tseb_scene / f"{date}_{level}_{satellite}_{LE}.tif"
        values = gu.get_raster_data(in_file)
        values[~valid] = np.nan
        le_ratio.append(values / sdn)

        in_file = tseb_scene / f"{date}_{level}_{satellite}_{LE_C}.tif"
        values = gu.get_raster_data(in_file)
        values[~valid] = np.nan
        le_c_ratio.append(values / sdn)

        in_file = tseb_scene / f"{date}_{level}_{satellite}_{LE_0}.tif"
        values = gu.get_raster_data(in_file)
        values[~valid] = np.nan
        le_0_ratio.append(values / sdn)

        in_file = tseb_scene / f"{date}_{level}_{satellite}_{LE_C_0}.tif"
        values = gu.get_raster_data(in_file)
        values[~valid] = np.nan
        le_c_0_ratio.append(values / sdn)

    if "SEN" in satellite:
        mission = satellite[:-1]
    elif "LCD" in satellite:
        mission = satellite[:-2]
    else:
        mission = satellite

    le_ratio = np.nanmean(np.asarray(le_ratio), axis=0)
    le_c_ratio = np.nanmean(np.asarray(le_c_ratio), axis=0)
    le_0_ratio = np.nanmean(np.asarray(le_0_ratio), axis=0)
    le_c_0_ratio = np.nanmean(np.asarray(le_c_0_ratio), axis=0)
    if is_high_resolution:
        fid = gu.resample_with_gdalwarp(sdn_daily_file,
                                        template_file,
                                        "bilinear",
                                        "MEM")
        sdn = fid.GetRasterBand(1).ReadAsArray()
        del fid
        fid = gu.resample_with_gdalwarp(ta_daily_file,
                                        template_file,
                                        "bilinear",
                                        "MEM")
        ta = fid.GetRasterBand(1).ReadAsArray()
        del fid
    else:
        sdn = gu.get_raster_data(sdn_daily_file)
        ta = gu.get_raster_data(ta_daily_file)

    output_file = tseb_dir / f"{date_str}_LEVEL4_{mission}_{ET}.tif"
    log.info(f"Compute Daily ET and saving to {output_file}")
    daily_flux = tseb.met.flux_2_evaporation(sdn * le_ratio, ta,
                                             time_domain=24)
    gu.save_image(daily_flux, gt, proj, output_file, no_data_value=np.nan)

    output_file = tseb_dir / f"{date_str}_LEVEL4_{mission}_{T}.tif"
    log.info(f"Compute Daily T and saving to {output_file}")
    daily_flux = tseb.met.flux_2_evaporation(sdn * le_c_ratio, ta,
                                             time_domain=24)
    gu.save_image(daily_flux, gt, proj, output_file, no_data_value=np.nan)

    output_file = tseb_dir / f"{date_str}_LEVEL4_{mission}_{ET_0}.tif"
    log.info(f"Compute Daily potential ET and saving to {output_file}")
    daily_flux = tseb.met.flux_2_evaporation(sdn * le_0_ratio, ta,
                                             time_domain=24)
    gu.save_image(daily_flux, gt, proj, output_file, no_data_value=np.nan)

    output_file = tseb_dir / f"{date_str}_LEVEL4_{mission}_{T_0}.tif"
    log.info(f"Compute Daily potential T and saving to {output_file}")
    daily_flux = tseb.met.flux_2_evaporation(sdn * le_c_0_ratio, ta,
                                             time_domain=24)
    gu.save_image(daily_flux, gt, proj, output_file, no_data_value=np.nan)
 
