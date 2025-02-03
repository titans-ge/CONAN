import numpy as np
from types import SimpleNamespace, FunctionType

default_LCpars_dict = dict( Duration     = 0.1,
                            rho_star     = None,
                            RpRs         = 0.,
                            Impact_para  = 0,
                            T_0          = 0,
                            Period       = 1,
                            Eccentricity = 0.,
                            omega        = 90,
                            q1           = 0,
                            q2           = 0,
                            D_occ        = 0,
                            Fn           = 0,
                            ph_off       = 0,
                            A_ev         = 0,
                            A_db         = 0
                            )

default_RVpars_dict = dict( T_0          = 0,
                            Period       = 1,
                            Eccentricity = 0.,
                            omega        = 90,
                            K            = 0
                            )

filter_shortcuts = dict(kepler='Kepler/Kepler.k',
                        tess='TESS/TESS.Red',
                        cheops='CHEOPS/CHEOPS.band',
                        wfc3_g141='HST/WFC3_IR.G141',
                        wfc3_g102='HST/WFC3_IR.G102',
                        sp36='Spitzer/IRAC.I1',
                        sp45='Spitzer/IRAC.I2',
                        ug='Geneva/Geneva.U',
                        b1='Geneva/Geneva.B1',
                        b2='Geneva/Geneva.B2',
                        bg='Geneva/Geneva.B',
                        gg='Geneva/Geneva.G',
                        v1='Geneva/Geneva.V2',
                        vg='Geneva/Geneva.V',
                        sdss_g='SLOAN/SDSS.g',
                        sdss_r='SLOAN/SDSS.r',
                        sdss_i='SLOAN/SDSS.i',
                        sdss_z='SLOAN/SDSS.z')

def _raise(exception_type, msg):
    raise exception_type(msg)

def _print_output(self, section: str, file=None):
    """function to print to screen/file the different sections of CONAN setup"""
    
    prior_str = lambda v: 'None' if v==None else f'F({v})' if isinstance(v,(int,float)) else f"N({v[0]},{v[1]})" if len(v)==2 else f"U({v[0]},{v[1]},{v[2]})" if len(v)==3 else f"TN({v[0]},{v[1]},{v[2]},{v[3]})"

    lc_possible_sections = ["lc_baseline", "sinusoid", "gp", "planet_parameters", "custom_LCfunction","depth_variation","timing_variation",
                            "phasecurve", "limb_darkening", "contamination","sinusoid"]
    rv_possible_sections = ["rv_baseline", "rv_gp", "custom_RVfunction"]
    fit_possible_sections = ["fit", "stellar_pars"]
    spacing = "" if file is None else "\t"

    if self._obj_type == "lc_obj":
        assert section in lc_possible_sections, f"{section} not a valid section of `lc_obj`. Section must be one of {lc_possible_sections}."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
        max_filt_len = max([len(n) for n in self._filters]+[len("flt")])  #max length of lc filter name
        max_wv_len   = max([len(str(n)) for n in self._wl]+[4])  #max length of wavelength

    if self._obj_type == "rv_obj":
        assert section in rv_possible_sections, f"{section} not a valid section of `rv_obj`. Section must be one of {rv_possible_sections}."
        max_name_len = max([len(n) for n in self._names]+[len("name")])      #max length of lc filename
    if self._obj_type == "fit_obj":
        assert section in fit_possible_sections, f"{section} not a valid section of `fit_obj`. Section must be one of {fit_possible_sections}."

    if section == "lc_baseline":
        _print_lc_baseline = """# ============ Input lightcurves, filters baseline function =======================================================""" +\
                            f"""\n{spacing}{"name":{max_name_len}s} {"flt":{max_filt_len}s} {"𝜆_𝜇m":{max_wv_len}s} |{"Ssmp":4s} {"ClipOutliers":12s} {"scl_col":7s} |{"off":3s} {"col0":4s} {"col3":4s} {"col4":4s} {"col5":4s} {"col6":4s} {"col7":4s} {"col8":4s}|{"sin":3s} {"id":2s} {"GP":2s} {"spline":15s}"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_name_len}s}} {{1:{max_filt_len}s}} {{2:{max_wv_len}s}}"+" |{3:4s} {4:12s} {5:7s} |{6:3s} {7:4d} {8:4d} {9:4d} {10:4d} {11:4d} {12:4d} {13:4d}|{14:3s} {15:2d} {16:2s} {17:15s}"        
        for i in range(len(self._names)):
            t = txtfmt.format(self._names[i], self._filters[i], str(self._wl[i]), self._ss[i].config,self._clipped_data.config[i], self._rescaled_data.config[i],
                                "  "+self._fit_offset[i], *self._bases[i][:-1], self._groups[i], self._useGPphot[i],self._lcspline[self._names[i]].conf, 
                                )
            _print_lc_baseline += t
        print(_print_lc_baseline, file=file)

    if section == "sinusoid":
        DA = self._sine_dict
        _print_sinusoid = f"""# ============ Sinusoidal signals: Amp*trig(2𝜋/P*(x-x0)) - trig=sin or cos or both added==========================""" +\
                            f"""\n{spacing}{"name/filt":{max_name_len}s} {"trig":7s}  {"n_harmonics":11s}  {"x":4s}  {"Amp[ppm]":18s}  {"P":18s}  {"x0":18s}"""
        #define print out format
        max_namefilt_len = max([len(n) for n in self._names+self._filters]+[9])      #max length of lcname/filtname
        txtfmt = f"\n{spacing}{{0:{max_namefilt_len}s}}"+" {1:7s}  {2:11d}  {3:4s}  {4:18s}  {5:18s}  {6:18s}"

        for k,v in DA.items():
            if v.trig is not None:
                amp_pri = f"F({v.Amp.user_input})" if isinstance(v.Amp.user_input, (float,int)) else f"N({v.Amp.user_input[0]},{v.Amp.user_input[1]})" if len(v.Amp.user_input)==2 else f"U({v.Amp.user_input[0]},{v.Amp.user_input[1]},{v.Amp.user_input[2]})"
                P_pri   = "None" if v.P.user_input==None else f"F({v.P.user_input})" if isinstance(v.P.user_input, (float,int)) else f"N({v.P.user_input[0]},{v.P.user_input[1]})" if len(v.P.user_input)==2 else f"U({v.P.user_input[0]},{v.P.user_input[1]},{v.P.user_input[2]})"
                x0_pri  = "None" if v.x0.user_input==None else f"F({v.x0.user_input})" if isinstance(v.x0.user_input, (float,int)) else f"N({v.x0.user_input[0]},{v.x0.user_input[1]})" if len(v.x0.user_input)==2 else f"U({v.x0.user_input[0]},{v.x0.user_input[1]},{v.x0.user_input[2]})"
                t = txtfmt.format(v.name, v.trig, v.n, v.par, amp_pri, P_pri, x0_pri)
                _print_sinusoid += t
        print(_print_sinusoid, file=file)


    if section == "gp":
        DA = self._GP_dict
        max_namefilt_len = max([len(n) for n in self._names+self._filters]+[9])      #max length of lcname/filtname
        _print_gp = f"""# ============ Photometry GP properties ==========================================================================="""
        _print_gp += f"""\n{spacing}{"name/filt":{max_namefilt_len}s} {'par1':4s} {"kern1":5s} {'Amplitude1_ppm':18s} {'length_scale':17s} |{'op':2s}| {'par2':4s} {"kern2":5s} {'Amplitude2_ppm':18s} {'length_scale2':17s}"""
        if DA != {}: 
            #define gp print out format
            txtfmt = f"\n{spacing}{{0:{max_namefilt_len}s}}"+" {1:4s} {2:5s} {3:18s} {4:17s} |{5:2s}| {6:4s} {7:5s} {8:18s} {9:17s} "        
            if self._sameLCgp.filtflag:
                for f in self._sameLCgp.filters:
                    lc = self._sameLCgp.LCs[f][0]

                    ngp = DA[lc]["ngp"]
                    if ngp == 2:
                        t = txtfmt.format(f, 
                                            DA[lc]["amplitude0"].user_data.col, DA[lc]["amplitude0"].user_data.kernel,  
                                            DA[lc]["amplitude0"].prior_str, DA[lc]["lengthscale0"].prior_str, DA[lc]["op"], 
                                            DA[lc]["amplitude1"].user_data.col, DA[lc]["amplitude1"].user_data.kernel,
                                            DA[lc]["amplitude1"].prior_str, DA[lc]["lengthscale1"].prior_str)
                    else:
                        t = txtfmt.format(f, 
                                            DA[lc]["amplitude0"].user_data.col, DA[lc]["amplitude0"].user_data.kernel,  
                                            DA[lc]["amplitude0"].prior_str, DA[lc]["lengthscale0"].prior_str, 
                                            "--", "None", "None", "None", "None")
                    _print_gp += t

            else:
                if self._allLCgp:  #shortcut print just one line gp config if all LCs have the same GP
                    equal_allgp = all([_compare_nested_structures(DA[list(DA.keys())[0]],DA[lc]) for lc in list(DA.keys())[1:]])
                else:
                    equal_allgp = False
                for lc in DA.keys():
                    ngp = DA[lc]["ngp"]
                    if ngp == 2:
                        t = txtfmt.format('same' if self._sameLCgp.flag else "all" if equal_allgp else lc, 
                                            DA[lc]["amplitude0"].user_data.col, DA[lc]["amplitude0"].user_data.kernel,  
                                            DA[lc]["amplitude0"].prior_str, DA[lc]["lengthscale0"].prior_str, DA[lc]["op"], 
                                            DA[lc]["amplitude1"].user_data.col, DA[lc]["amplitude1"].user_data.kernel,
                                            DA[lc]["amplitude1"].prior_str, DA[lc]["lengthscale1"].prior_str)
                    else:
                        t = txtfmt.format('same' if self._sameLCgp.flag else "all" if equal_allgp else lc, 
                                            DA[lc]["amplitude0"].user_data.col, DA[lc]["amplitude0"].user_data.kernel,  
                                            DA[lc]["amplitude0"].prior_str, DA[lc]["lengthscale0"].prior_str, 
                                            "--", "None", "None", "None", "None")
                    _print_gp += t
                    if self._sameLCgp.flag or equal_allgp:      #dont print the other lc GPs if same_GP is True
                        break
        print(_print_gp, file=file)

    if section == "planet_parameters":
        DA = self._planet_pars
        notes = dict(RpRs="#range[-0.5,0.5]",Impact_para="#range[0,2]",K="#unit(same as RVdata)",T_0="#unit(days)",Period="#range[0,inf]days",
                        Eccentricity="#choice in []|range[0,1]/range[-1,1]",omega="#choice in []|range[0,360]deg/range[-1,1]",
                        sesinw="#choice in []|range[0,1]/range[-1,1]",secosw="#choice in []|range[0,360]deg/range[-1,1]")
        _print_planet_parameters = f"""# ============ Planet parameters (Transit and RV) setup ========================================================== """+\
                                    f"""\n{spacing}{'name':25s}  {'fit':3s} \t{'prior':35s}\tnote"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:25s}  {1:3s} \t{2:35s}\t{3}"
        #print line for stellar density or duration
        p    = "rho_star" if "rho_star" in DA[f'pl{1}'].keys() else "Duration"
        popt = "[rho_star]/Duration" if "rho_star" in DA[f'pl{1}'].keys() else "rho_star/[Duration]"
        _print_planet_parameters +=  txtfmt.format( popt, DA[f'pl{1}'][p].to_fit, DA[f'pl{1}'][p].prior_str, "#choice in []|unit(gcm^-3/days)")
        _print_planet_parameters +=  f"\n{spacing}--------repeat this line & params below for multisystem, adding '_planet_number' to the names e.g RpRs_1 for planet 1, ..."
        #then cycle through parameters for each planet       
        for n in range(1,self._nplanet+1):   
            lbl = f"_{n}" if self._nplanet>1 else ""
            ecc_w_opt = {}  
            ecc_w_opt["Eccentricity"] = ecc_w_opt["sesinw"] = f"Eccentricity{lbl}/[sesinw{lbl}]" if "sesinw" in DA[f'pl{1}'].keys() else f"[Eccentricity{lbl}]/sesinw{lbl}"
            ecc_w_opt["omega"]        = ecc_w_opt["secosw"] = f"omega{lbl}/[secosw{lbl}]" if "secosw" in DA[f'pl{1}'].keys() else f"[omega{lbl}]/secosw{lbl}"

            for i,p in enumerate(self._TR_RV_parnames):
                if p in ["rho_star","Duration"]: 
                    continue
                elif p in ["Eccentricity","omega","sesinw","secosw"]:
                    t = txtfmt.format( ecc_w_opt[p], DA[f'pl{n}'][p].to_fit, DA[f'pl{n}'][p].prior_str, notes[p])
                else:
                    t = txtfmt.format(  p+lbl, DA[f'pl{n}'][p].to_fit, DA[f'pl{n}'][p].prior_str, notes[p])
                _print_planet_parameters += t
            if n!=self._nplanet: _print_planet_parameters += f"\n{spacing}------------"
        print(_print_planet_parameters, file=file)

    if section == "custom_LCfunction":
        DA = self._custom_LCfunc
        flag = False if DA.func is None else True
        _print_custom_function = f"""# ============ Custom LC function (read from custom_LCfunc.py file)================================================"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:16s}}: {{1:40s}}\t{{2}}"
        _print_custom_function += txtfmt.format("function", DA.func.__name__ if flag else 'None', "#custom function/class to combine with/replace LCmodel")
        _print_custom_function += txtfmt.format("x",DA.x if flag else 'None',"#independent variable [time, phase_angle]")
        if flag:
            fa      = DA.func_args
            fa_str  = []
            for k in fa.keys():
                if isinstance(fa[k],(int,float)):
                    fa_str.append(f'{k}:F({fa[k]})')
                if isinstance(fa[k],tuple):
                    fa_str.append(f"{k}:{'U' if len(fa[k])==3 else 'N' if len(fa[k])==2 else 'TN'}{str(fa[k]).replace(' ','')}" )
            fa_str  = ",".join(fa_str)
        else: fa_str = 'None'
        _print_custom_function += txtfmt.format("func_pars",fa_str,"#param names&priors e.g. A:U(0,1,2),P:N(2,1)")
        exa_str = [f"{k}:{v}" for k,v in DA.extra_args.items()]
        exa_str = ",".join(exa_str) if exa_str!=[] else 'None'
        _print_custom_function += txtfmt.format("extra_args",exa_str,"#extra args to func as a dict e.g ld_law:quad")
        _print_custom_function += txtfmt.format("op_func",'None' if (DA.replace_LCmodel or not flag) else DA.op_func.__name__ ,"#function to combine the LC and custom models")
        _print_custom_function += txtfmt.format("replace_LCmodel",str(DA.replace_LCmodel) if flag else 'False',"#if the custom function replaces the LC model")
        print(_print_custom_function, file=file)

    if section == "custom_RVfunction":
        DA = self._custom_RVfunc
        flag = False if DA.func is None else True
        _print_custom_function = f"""# ============ Custom RV function (read from custom_RVfunc.py file)================================================"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:16s}}: {{1:40s}}\t{{2}}"
        _print_custom_function += txtfmt.format("function", DA.func.__name__ if flag else 'None', "#custom function/class to combine with/replace RVmodel")
        _print_custom_function += txtfmt.format("x",DA.x if flag else 'None',"#independent variable [time, true_anomaly]")
        if flag:
            fa      = DA.func_args
            fa_str  = []
            for k in fa.keys():
                if isinstance(fa[k],(int,float)):
                    fa_str.append(f'{k}:F({fa[k]})')
                if isinstance(fa[k],tuple):
                    fa_str.append(f"{k}:{'U' if len(fa[k])==3 else 'N' if len(fa[k])==2 else 'TN'}{str(fa[k]).replace(' ','')}" )
            fa_str  = ",".join(fa_str)
        else: fa_str = 'None'
        _print_custom_function += txtfmt.format("func_pars",fa_str,"#param names&priors e.g. A:U(0,1,2),P:N(2,1)")
        exa_str = [f"{k}:{v}" for k,v in DA.extra_args.items()]
        exa_str = ",".join(exa_str) if exa_str!=[] else 'None'
        _print_custom_function += txtfmt.format("extra_args",exa_str,"#extra args to func as a dict")
        _print_custom_function += txtfmt.format("op_func",'None' if (DA.replace_RVmodel or not flag) else DA.op_func.__name__ ,"#function to combine the RV and custom models")
        _print_custom_function += txtfmt.format("replace_RVmodel",str(DA.replace_RVmodel) if flag else 'False',"#if the custom function replaces the RV model")
        print(_print_custom_function, file=file)


    if section == "depth_variation":
        grnames    = np.array(list(sorted(set(self._groups))))
        ngroup     = len(grnames)
        _print_depth_variation = f"""# ============ ddF setup ========================================================================================"""+\
                                    f"""\n{spacing}{"Fit_ddFs":8s}\t{"dRpRs":16s}\tdiv_white"""

        #define print out format
        txtfmt = f"\n{spacing}"+"{0:8s}\t{1:16s}\t{2:3s}"        
        # pri_ddf = f"N({self._ddfs.drprs.prior_mean},{self._ddfs.drprs.prior_width_lo})" if self._ddfs.drprs.prior=="p" else f"U({self._ddfs.drprs.bounds_lo},{self._ddfs.drprs.start_value},{self._ddfs.drprs.bounds_hi})"
        pri_ddf = prior_str(self._ddfs.drprs.user_input)
        t = txtfmt.format(self._ddfs.ddfYN, pri_ddf, self._ddfs.divwhite)
        _print_depth_variation += t

        print(_print_depth_variation, file=file)

    if section == "timing_variation":
        _print_timing_variation = f"""# ============ TTV setup ========================================================================================"""+\
                                    f"""\n{spacing}{"Fit_TTVs":8s}\t{"dt_priors(deviation from linear T0)":35s}\t{"transit_baseline[P]":19s}\t{"per_LC_T0":10s}\tinclude_partial"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:8s}\t{1:35s}\t{2:19.4f}\t{3:10s}\t{4}"
        pri_ttv = f"N{self._ttvs.dt}" if len(self._ttvs.dt)==2 else f"U{self._ttvs.dt}"
        t = txtfmt.format(self._ttvs.to_fit, pri_ttv.replace(" ",""),self._ttvs.baseline, str(self._ttvs.per_LC_T0), str(self._ttvs.include_partial))
        _print_timing_variation += t
        print(_print_timing_variation, file=file)

    if section == "phasecurve":
        pars  = ["D_occ", "Fn", "ph_off","A_ev","A_db"]
        # descr = ["occultation depth", "atmospheric amplitude", "phase offset in degrees","ellipsoidal variation"]
        _print_phasecurve = f"""# ============ Phase curve setup ================================================================================ """+\
                                f"""\n{spacing}{'filt':{max_filt_len}s}  {'D_occ[ppm]':20s} {'Fn[ppm]':20s} {'ph_off[deg]':20s} {'A_ev[ppm]':20s} {'A_db[ppm]':20s}"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_filt_len}s}}"+"  {1:20s} {2:20s} {3:20s} {4:20s} {5:20s}"       
        
        DA = self._PC_dict
        for i,f in enumerate(self._filnames):
            pri_Docc  = prior_str(DA['D_occ'][f].user_input)
            pri_Fn    = prior_str(DA['Fn'][f].user_input)
            pri_phoff = prior_str(DA['ph_off'][f].user_input)
            pri_Aev   = prior_str(DA['A_ev'][f].user_input)
            pri_Adb   = prior_str(DA['A_db'][f].user_input)
            
            t = txtfmt.format(f, pri_Docc, pri_Fn, pri_phoff, pri_Aev, pri_Adb)
            _print_phasecurve += t
        print(_print_phasecurve, file=file)

    if section == "limb_darkening":
        DA = self._ld_dict
        _print_limb_darkening = f"""# ============ Limb darkening setup ============================================================================= """+\
                                f"""\n{spacing}{'filters':7s}\tfit\t{'q1':17s}\t{'q2':17s}"""
        #define print out format
        txtfmt = f"\n{spacing}"+"{0:7s}\t{1:3s}\t{2:17s}\t{3:17s}"       
        for i in range(len(self._filnames)):
            pri_q1 = f"N({DA['q1'][i]},{DA['sig_lo1'][i]})" if DA['sig_lo1'][i] else f"U({DA['bound_lo1'][i]},{DA['q1'][i]},{DA['bound_hi1'][i]})"  if DA['bound_hi1'][i] else f"F({DA['q1'][i]})"
            pri_q2 = f"N({DA['q2'][i]},{DA['sig_lo2'][i]})" if DA['sig_lo2'][i] else f"U({DA['bound_lo2'][i]},{DA['q2'][i]},{DA['bound_hi2'][i]})" if DA['bound_hi2'][i] else f"F({DA['q2'][i]})"
            to_fit = "y" if (pri_q1[0]!="F" or pri_q2[0]!="F") else "n"
            t = txtfmt.format(self._filnames[i], to_fit, pri_q1, pri_q2) 
            _print_limb_darkening += t

        print(_print_limb_darkening, file=file)

    if section == "contamination":
        DA = self._contfact_dict
        _print_contamination = f"""# ============ contamination setup (give contamination as flux ratio) ======================================== """+\
                                f"""\n{spacing}{'filters':7s}\tcontam_factor"""
        #define print out format
        txtfmt = f"\n{spacing}{{0:{max_filt_len}s}}"+"\t{1:20s}"       
        for i,f in enumerate(self._filnames):
            t = txtfmt.format(f,prior_str(DA[f].user_input))
            _print_contamination += t
        print(_print_contamination, file=file)

    if section == "stellar_pars":
        DA = self._stellar_dict
        _print_stellar_pars = f"""# ============ Stellar input properties ======================================================================"""+\
        f"""\n{spacing}{'# parameter':13s}   value """+\
        f"""\n{spacing}{'Radius_[Rsun]':13s}  N({DA['R_st'][0]},{DA['R_st'][1]})"""+\
        f"""\n{spacing}{'Mass_[Msun]':13s}  N({DA['M_st'][0]},{DA['M_st'][1]})"""+\
            f"""\n{spacing}Input_method:[R+rho(Rrho), M+rho(Mrho)]: {DA['par_input']}"""
        print(_print_stellar_pars, file=file)           

    if section == "fit":
        DA = self._fit_dict
        # if all DA['apply_LCjitter'] is the same, set it to one value
        app_jitt   = DA['apply_LCjitter'][0] if len(set(DA['apply_LCjitter']))==1 else str(DA['apply_LCjitter']).replace(" ","")
        app_RVjitt = DA['apply_RVjitter'][0] if len(set(DA['apply_RVjitter']))==1 else str(DA['apply_RVjitter']).replace(" ","")

        _print_fit_pars = f"""# ============ FIT setup ====================================================================================="""+\
        f"""\n{spacing}{'Number_steps':40s}  {DA['n_steps']} \n{spacing}{'Number_chains':40s}  {DA['n_chains']} \n{spacing}{'Number_of_processes':40s}  {DA['n_cpus']} """+\
            f"""\n{spacing}{'Burnin_length':40s}  {DA['n_burn']} \n{spacing}{'n_live':40s}  {DA['n_live']} \n{spacing}{'force_nlive':40s}  {DA['force_nlive']} \n{spacing}{'d_logz':40s}  {DA['dyn_dlogz']} """+\
                    f"""\n{spacing}{'Sampler(emcee/dynesty)':40s}  {DA['sampler']} \n{spacing}{'emcee_move(stretch/demc/snooker)':40s}  {DA['emcee_move']} """+\
                    f"""\n{spacing}{'nested_sampling(static/dynamic[pfrac])':40s}  {DA['nested_sampling']} \n{spacing}{'leastsq_for_basepar(y/n)':40s}  {DA['leastsq_for_basepar']} """+\
                        f"""\n{spacing}{'apply_LCjitter(y/n,list)':40s}  {app_jitt} \n{spacing}{'apply_RVjitter(y/n,list)':40s}  {app_RVjitt} """+\
                            f"""\n{spacing}{'LCjitter_loglims(auto/[lo,hi])':40s}  {str(DA['LCjitter_loglims']).replace(" ","")} \n{spacing}{'RVjitter_lims(auto/[lo,hi])':40s}  {str(DA['RVjitter_lims']).replace(" ","")} """+\
                                f"""\n{spacing}{'LCbasecoeff_lims(auto/[lo,hi])':40s}  {str(DA['LCbasecoeff_lims']).replace(" ","")} \n{spacing}{'RVbasecoeff_lims(auto/[lo,hi])':40s}  {str(DA['RVbasecoeff_lims']).replace(" ","")} """+\
                                    f"""\n{spacing}{'Light_Travel_Time_correction(y/n)':40s}  {DA['LTT_corr']}""" # \n{spacing}{'fit_LCoffset(y/n or list)':40s}  {fit_off}

        
        print(_print_fit_pars, file=file)

    if section == "rv_baseline":
        _print_rv_baseline = """# ============ Input RV curves, baseline function, GP, spline,  gamma ============================================ """+\
                                f"""\n{spacing}{'name':{max_name_len}s} {'RVunit':6s} {"scl_col":7s} |{'col0':4s} {'col3':4s} {'col4':4s} {"col5":4s}| {'sin':3s} {"GP":2s} {"spline_config  ":15s} | {f'gamma_{self._RVunit}':14s} """
        if self._names != []:
            DA = self._rvdict
            txtfmt = f"\n{spacing}{{0:{max_name_len}s}}"+" {1:6s} {2:7s} |{3:4d} {4:4d} {5:4d} {6:4d}| {7:3d} {8:2s} {9:15s} | {10:14s}"         
            for i in range(self._nRV):
                # gam_pri_ = f'N({DA["gammas"][i]},{DA["sig_lo"][i]})' if DA["sig_lo"][i] else f'U({DA["bound_lo"][i]},{DA["gammas"][i]},{DA["bound_hi"][i]})' if DA["bound_hi"][i] else f"F({DA['gammas'][i]})"
                gam_pri_ = prior_str(DA["gamma"][i].user_input)
                t = txtfmt.format(self._names[i],self._RVunit,self._rescaled_data.config[i], *self._RVbases[i],
                                    self._useGPrv[i],self._rvspline[self._names[i]].conf,gam_pri_)
                _print_rv_baseline += t
        print(_print_rv_baseline, file=file)

    if section == "rv_gp":
        DA = self._rvGP_dict
        _print_gp = f"""# ============ RV GP properties ================================================================================== """
        # _print_gp += f"""\nsame_GP: {self._sameRVgp.flag}"""
        _print_gp += f"""\n{spacing}{"name":{max_name_len}s} {'par1':4s} {"kern1":5s} {'Amplitude1':18s} {'length_scale':17s} |{'op':2s}| {'par2':4s} {"kern2":5s} {'Amplitude2':18s} {'length_scale2':15s}"""
        if DA != {}: 
            #define gp print out format
            txtfmt = f"\n{spacing}{{0:{max_name_len}s}}"+" {1:4s} {2:5s} {3:18s} {4:17s} |{5:2s}| {6:4s} {7:5s} {8:18s} {9:15s} "        

            for rv in DA.keys():
                ngp = DA[rv]["ngp"]
                if ngp == 2:
                    t = txtfmt.format('same' if self._sameRVgp.flag else rv,
                                        DA[rv]["amplitude0"].user_data.col, DA[rv]["amplitude0"].user_data.kernel,  
                                        DA[rv]["amplitude0"].prior_str,  DA[rv]["lengthscale0"].prior_str, DA[rv]["op"], 
                                        DA[rv]["amplitude1"].user_data.col, DA[rv]["amplitude1"].user_data.kernel,
                                        DA[rv]["amplitude1"].prior_str, DA[rv]["lengthscale1"].prior_str)
                else:
                    t = txtfmt.format('same' if self._sameRVgp.flag else rv,DA[rv]["amplitude0"].user_data.col, 
                                        DA[rv]["amplitude0"].user_data.kernel, DA[rv]["amplitude0"].prior_str,
                                        DA[rv]["lengthscale0"].prior_str, "--", "None", "None", "None", "None")
                _print_gp += t
                if self._sameRVgp.flag:      #dont print the other GPs if same_GP is True
                    break
        print(_print_gp, file=file)

class _param_obj():
    def __init__(self,to_fit,start_value,step_size,
                    prior, prior_mean, prior_width_lo, prior_width_hi,
                    bounds_lo, bounds_hi,user_input=None,user_data=None,prior_str=None):
        """  
        convenience class to create a parameter object with the following Attributes

        Parameters:
        -----------
        to_fit : str;
            'y' or 'n' to fit or not fit the parameter.
        start_value : float;
            starting value for the parameter.
        step_size : float;
            step size for the parameter.
        prior : str;
            'n' or 'p' to not use (n) or use (p) a normal prior.
        prior_mean : float;
            mean of the normal prior.
        prior_width_lo : float;
            lower sigma of the normal prior.
        prior_width_hi : float;
            upper sigma of the normal prior.
        bounds_lo : float;
            lower bound for the parameter.
        bounds_hi : float;
            upper bound for the parameter.
        user_data : any;
            any data to be stored in the parameter object.
        user_info: tuple, int, float:
            stores prior input given by the user
        prior_str : str;
            string representation of the prior for printing.

        Returns:
        -----------
        param_obj : object;
            object with the parameters.
        """
    
        self.to_fit         = to_fit if (to_fit in ["n","y"]) else _raise(ValueError, "to_fit (to_fit) must be 'n' or 'y'")
        self.start_value    = start_value
        self.step_size      = step_size
        self.prior          = prior if (prior in ["n","p"]) else _raise(ValueError, "prior (prior) must be 'n' or 'p'")
        self.prior_mean     = prior_mean
        self.prior_width_lo = prior_width_lo
        self.prior_width_hi = prior_width_hi
        self.bounds_lo      = bounds_lo
        self.bounds_hi      = bounds_hi
        self.user_input     = user_input
        self.user_data      = user_data
        self.prior_str      = prior_str

    @classmethod
    def from_tuple(cls, param_in, step=None,lo=None, hi=None, user_input=None,user_data=None,func_call=""):
        """
        alternative method to initialize _param_obj using from a tuple.
        * if int/float is given returns: (to_fit="n",start_value=param_in,step_size=0,prior="n",prior_mean=param_in,prior_width_lo=0,prior_width_hi=0,bounds_lo=0,bounds_hi=0,user_input=None)
        * if tuple of len 2 it returns:  (to_fit="y",start_value=param_in[0],step_size=0.1*param_in[1],prior="p",prior_mean=param_in[0],prior_width_lo=param_in[1],prior_width_hi=param_in[1],bounds_lo=param_in[0]-10*param_in[1],bounds_hi=param_in[0]+10*param_in[1],user_input=None)
        * if tuple of len 3 it returns:  (to_fit="y",start_value=param_in[0],step_size=0.001*np.ptp(param_in),prior="n",prior_mean=param_in[0],prior_width_lo=0,prior_width_hi=0,bounds_lo=param_in[0],bounds_hi=param_in[2],user_input=None)
        * if tuple of len 4 it returns:  (to_fit="y",start_value=param_in[2],step_size=0.1*param_in[3],prior="p",prior_mean=param_in[2],prior_width_lo=param_in[3],prior_width_hi=param_in[3],bounds_lo=param_in[0],bounds_hi=param_in[1],user_input=None)

        Parameters:
        -----------
        param_in : int, float,tuple,None;
            input float/tuple with the parameters for the object.
        step : float,None;
            step size for the parameter to override the default value derived from param
        lo : float,None;
            lower bound for the parameter to override the default value derived from param_in
        hi : float;
            upper bound for the parameter to override the default value derived from param_in
        user_input; tuple, int, float:
            stores prior input given by the user
        user_data : any;
            any data to be stored in the parameter object.
        func_call : str;
            name of the function calling this method, to be used in error messages.
    
        Returns:
        -----------
        param_obj : object;
            object with the parameters.

        Example:
        -----------
        >>> RpRs = (0.1,0.002)
        >>> param_obj = _param_obj.from_tuple(RpRs,func_call="planet_parameters():") 
        """
        assert isinstance(func_call,str),f"_param_obj.from_tuple() func_call must be a string but {func_call} given."
        
        v = param_in
        if isinstance(v, (int, float)): #fixed parameter
            params = ["n",v,0.,"n",v,0.,0.,0.,0.,user_input,user_data,f'F({v})']
        elif isinstance(v, tuple):
            if len(v)==2:  #normal prior
                step   = 0.1*v[1] if step==None else step
                lo_lim = v[0]-10*v[1] if lo==None else lo
                hi_lim = v[0]+10*v[1] if hi==None else hi
                params = ["y",v[0],step,"p",v[0],v[1],v[1],lo_lim,hi_lim,user_input,user_data,f'N({v[0]},{v[1]})']
            elif len(v)==3: #uniform prior
                assert v[0]<=v[1]<=v[2],f"{func_call} wrongly defined uniform prior. must be of form (min,start,max) with min<=start<=max but {v} given."
                step = min(0.001,0.001*np.ptp(v)) if step==None else step
                lo_lim = v[0] if lo==None else lo
                hi_lim = v[2] if hi==None else hi
                params = ["y",v[1],step,"n",v[1],0,0,lo_lim,hi_lim,user_input,user_data,f'U({v[0]},{v[1]},{v[2]})']
            elif len(v)==4: #truncated normal prior
                assert v[0]<=v[2]<=v[1],f"{func_call} wrongly defined trucated normal prior. must be of form (min,max,mean,std) with min<=mean<=max but {v} given."
                step = 0.1*v[3] if step==None else step
                params = ["y",v[2],step,"p",v[2],v[3],v[3],v[0],v[1],user_input,user_data,f'TN({v[0]},{v[1]},{v[2]},{v[3]})']
            else:
                raise TypeError(f"{func_call} tuple must have 2,3 or 4 elements but {v} given")
        elif v==None:
            params = ["n",None,0,"n",None,0,0,0,0,user_input,user_data,'None']
        else:
            raise TypeError(f"{func_call} input must be an int, float, tuple or None")
        return cls(*params)

    def _set(self, par_list):
        return self.__init__(*par_list)
    
    def __repr__(self):
        return f"{self.__dict__}"
    
    def _get_list(self):
        return [p for p in self.__dict__.values()]

class _text_format:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'



def _compare_nested_structures(obj1, obj2, verbose=False):
    """  
    Compare two nested structures (e.g. dictionaries, lists, etc.) for equality.
    """
    
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            if verbose: print(f"keys differ in {set(obj1) - set(obj2)}")
            return False
        return all(_compare_nested_structures(obj1[key], obj2[key]) for key in obj1)
    
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            return False
        return all([_compare_nested_structures(item1, item2) for item1, item2 in zip(obj1, obj2)])
    
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        return np.array_equal(obj1, obj2)
    
    elif isinstance(obj1, SimpleNamespace) and isinstance(obj2, SimpleNamespace):
        return all([_compare_nested_structures(vars(obj1)[key], vars(obj2)[key]) for key in vars(obj1)])
    

    elif isinstance(obj1, FunctionType) and isinstance(obj2, FunctionType):
        return (obj1.__code__.co_code == obj2.__code__.co_code and
                obj1.__code__.co_consts == obj2.__code__.co_consts and
                obj1.__code__.co_names == obj2.__code__.co_names and
                obj1.__code__.co_varnames == obj2.__code__.co_varnames)

    elif ("CONAN" in str(type(obj1))) and ("CONAN" in str(type(obj1))):
        return all([_compare_nested_structures(vars(obj1)[key], vars(obj2)[key]) for key in vars(obj1)])
    
    else:
        return obj1 == obj2


def compare_objs(obj1,obj2):
    """   
    compare two objects for equality
    """
    res = _compare_nested_structures(obj1,obj2)
    if res:
        return True
    else: 
        for k,v in obj1.__dict__.items():
            res = _compare_nested_structures(obj1.__dict__[k], obj2.__dict__[k])
            if not res: print(f"{k:25s}: {res}")
        return False