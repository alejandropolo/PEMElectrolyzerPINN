import numpy as np
# TODO: Add a python library to work with units
class ELCellStack:
    def __init__(self):
        # Input parameters
        self.R = 8.314  # Ideal gas constant [J/mol/K]
        self.F = 96485  # Faraday's constant [C/mol]

        self.A = 680  # Area of cell [cm^2] [Thesis]
        # self.A = 2.89  # Area of cell [cm^2] [Liso18]

        self.Alpha_an = 0.5  # [Thesis] Transfer coefficient
        self.Alpha_cat = 0.5  # [Thesis] Transfer coefficient
        # self.Alpha_an = 1.2  # [Liso18] Transfer coefficient
        # self.Alpha_cat = 0.5  # [Liso18] Transfer coefficient

        self.il = 6  # Limiting current density [A/cm^2]
        self.initial_thickness = 1.78e-2  # Initial membrane thickness [cm] [Thesis]
        self.lm = self.initial_thickness  # Membrane thickness [cm] [Thesis]
        # self.lm = 1.75e-2  # Membrane thickness [cm] [Liso18]

        self.lambdam = 20  # Membrane hydration parameter

        self.roughness_an = 7.23e2  # Roughness factor anode [cm^2/cm^2]
        self.roughness_cat = 2.33e2  # Roughness factor cathode [cm^2/cm^2]

        self.i0ref_an = 2.3e-7  # [Thesis] Anode exchange current density at ref temperature [A/cm^2]
        self.i0ref_cat = 1e-3  # [Thesis] Cathode exchange current density at ref temperature [A/cm^2]
        # self.i0ref_an = 5e-12  # [Liso18] Anode exchange current density at ref temperature [A/cm^2]
        # self.i0ref_cat = 1e-3  # [Liso18] Cathode exchange current density at ref temperature [A/cm^2]

        self.vthn = 1.481  # Thermoneutral voltage [V]
        self.MmH2O = 18  # Water molar mass [g/mol]
        self.rhoH2O = 997  # Water density [kg/m^3]
        self.Ea_an = 76000  # Anode activation energy [J/mol]
        self.Ea_cat = 4300  # Cathode activation energy [J/mol]
        self.Tref = 298  # Reference temperature [K]

    def VCell(self, Tk, i_cell, pres):
        pcat = pres  # Pressure at cathode [bar]
        pan = pres  # Pressure at anode [bar]
        ppH2 = pan - self.PsatH2O(Tk)
        ppO2 = pcat - self.PsatH2O(Tk)
        E = 1.229 - 0.9e-3 * (Tk - 298)
        i0_an = self.roughness_an * self.i0ref_an * np.exp(-(self.Ea_an / self.R) * (1 / Tk - 1 / 353))
        i0_cat = self.roughness_cat * self.i0ref_cat * np.exp(-(self.Ea_cat / self.R) * (1 / Tk - 1 / 353))
        #TODO: Generalize for the case when c is not equal in anode and cathode
        c = self.R * Tk / (self.Alpha_an * self.F)


        k1_nernst = E - (self.R*Tk)/(2*self.F)*np.log(self.PsatH2O(Tk) / (ppH2 * (ppO2 ** 0.5)))
        k1_act = c * (-np.log(i0_an)-np.log(i0_cat))
        k1 = k1_nernst + k1_act

        k2 = 2*self.R * Tk / (self.Alpha_an * self.F)
        k3 = self.lm * 1 / ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))

        if i_cell > 0:
            V_cell = k1 + k2 * np.log(i_cell) + k3 * i_cell
        else:
            V_cell = k1_nernst
        return V_cell, k1_nernst, k1_act, k2, k3
    
    def VCell_deg(self, Tk, i_cell, pres,lm):
        pcat = pres  # Pressure at cathode [bar]
        pan = pres  # Pressure at anode [bar]
        ppH2 = pan - self.PsatH2O(Tk)
        ppO2 = pcat - self.PsatH2O(Tk)
        E = 1.229 - 0.9e-3 * (Tk - 298)
        EW = 1.100  # Nafion equivalent weight [kg/mol]
        w = self.lambdam * self.MmH2O * 10 ** (-3) / EW  # Normalized water uptake
        SH2 = 1 / (w * 1.09e5 * np.exp(77 / Tk)) * 10 ** 5  # Hydrogen solubility in water [mol/m^3/bar]
        DH2 = 1.23e-6 * np.exp(-(-2602) / Tk)  # Hydrogen diffusivity in water [m^2/s]
        ppH2 = pres - self.PsatH2O(Tk)  # Hydrogen partial pressure [bar]
        iH2 = SH2 * DH2 * ppH2 / lm * 2 * self.F  # gas crossover current [A/cm^2]

       


        i0_an = self.roughness_an * self.i0ref_an * np.exp(-(self.Ea_an / self.R) * (1 / Tk - 1 / 353))
        i0_cat = self.roughness_cat * self.i0ref_cat * np.exp(-(self.Ea_cat / self.R) * (1 / Tk - 1 / 353))
        # nx = self.R * Tk * np.log(iH2 / i0_cat) / self.F  # gas crossover overvoltage [V]
        #TODO: Generalize for the case when c is not equal in anode and cathode
        c = self.R * Tk / (self.Alpha_an * self.F)

        ## TODO: Include again nx term (it has been omited because of its dependence to the membrane thickness to simply the final equation)
        # k1_nernst = E - (self.R*Tk)/(2*self.F)*np.log(self.PsatH2O(Tk) / (ppH2 * (ppO2 ** 0.5))) + nx
        k1_nernst = E - (self.R*Tk)/(2*self.F)*np.log(self.PsatH2O(Tk) / (ppH2 * (ppO2 ** 0.5)))
        k1_act = c * (-np.log(i0_an)-np.log(i0_cat))
        k1 = k1_nernst + k1_act

        k2 = 2*self.R * Tk / (self.Alpha_an * self.F)
        # k3 = 1 / ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))
        sigma = ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))
        k3 = (self.initial_thickness**2) / sigma
        if i_cell > 0:
            V_cell = k1 + k2 * np.log(i_cell) + k3 * (1/lm)  * i_cell
        else:
            V_cell = k1_nernst
        return V_cell, k1, k2, k3
    
    def ENernst(self, Tk, pres):
        pcat = pres  # Pressure at cathode [bar]
        pan = pres  # Pressure at anode [bar]
        ppH2 = pan - self.PsatH2O(Tk)
        ppO2 = pcat - self.PsatH2O(Tk)
        E = 1.229 - 0.9e-3 * (Tk - 298)
        Gf_liq = E * 2 * self.F
        ENernst = Gf_liq / (2 * self.F) - ((self.R * Tk) * np.log(self.PsatH2O(Tk) / (ppH2 * (ppO2 ** 0.5)))) / (2 * self.F)
        return ENernst

    def VAct(self, Tk, i_cell):
        i0_an = self.roughness_an * self.i0ref_an * np.exp(-(self.Ea_an / self.R) * (1 / Tk - 1 / 353))
        i0_cat = self.roughness_cat * self.i0ref_cat * np.exp(-(self.Ea_cat / self.R) * (1 / Tk - 1 / 353))
        c_an = self.R * Tk / (self.Alpha_an * self.F)
        c_cat = self.R * Tk / (self.Alpha_cat * self.F)

        ## TODO: Substitute archsinh by log
        b1 = np.arcsinh(i_cell/ (2 * i0_an))
        b2 = np.arcsinh(i_cell/ (2 * i0_cat))
        Vact = c_an * b1 + c_cat * b2
        return Vact
    
    def VAct_log(self, Tk, i_cell):
        i0_an = self.roughness_an * self.i0ref_an * np.exp(-(self.Ea_an / self.R) * (1 / Tk - 1 / 353))
        i0_cat = self.roughness_cat * self.i0ref_cat * np.exp(-(self.Ea_cat / self.R) * (1 / Tk - 1 / 353))
        c_an = self.R * Tk / (self.Alpha_an * self.F)
        c_cat = self.R * Tk / (self.Alpha_cat * self.F)

        if i_cell == 0:
            b1 = 0
            b2 = 0
        else:
            b1 = np.log(i_cell / i0_an)
            b2 = np.log(i_cell / i0_cat)
        Vact = c_an * b1 + c_cat * b2
        return Vact

    def VOhm(self, Tk, i_cell):
        # Area Specific ohmic Resistance
        r = self.lm * 1 / ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))  # [Ohm*cm^2]
        VOhm = (i_cell * r)  # [V]
        return VOhm

    def VConc(self, Tk, i_cell):
        VConc = self.R * Tk / (2 * self.F) * (1 + 1 / self.Alpha_cat) * np.log(self.il / (self.il - i_cell))  # [V]
        return VConc

    def PsatH2O(self, Tk):
        # Compute the saturated water vapor pressure
        Tc = Tk - 273.15  # Convert to celsius
        PsatH2O = 0.0061 * np.exp((Tc / (Tc + 238.3)) * 17.2694)  # Pressure in bar
        return PsatH2O

    def conc(self, Tk, i_cell, pres):
        ## TODO: Revisar por qué se hace esta division (es como si hubiese que pasarlo a A/m^2)
        i_cell = i_cell / (10 ** (-4))  # Corrent density [A/cm^2]
        eta = 0.695  # Equilibrium overpotential 2e-ORR [V]
        pO2 = pres - self.PsatH2O(Tk)  # Oxygen partial pressure [bar]
        sO2 = 1.62e-6 * np.exp(603 / Tk) * 10 ** 5  # Oxygen solubility [mol/m^3/bar]
        cO2 = sO2 * pO2  # Oxygen concentration [mol/m^3]
        Qc = self.MmH2O * i_cell * (self.A * 10 ** (-4)) / (2 * self.F * self.rhoH2O * 10 ** 3)  # Consumed waterflow [m^3/s]
        Qt = (-0.332 * np.log(i_cell) + 5.59) * Qc  # Transferred waterflow [m^3/s]
        vH2O = Qt / (self.A * 10 ** (-4))  # Water velocity [m/s]
        EW = 1.100  # Nafion equivalent weight [kg/mol]
        rhonaf = 1980  # Nafion dry membrane density [kg/m^3]
        Cmemb = rhonaf / EW  # Membrane concentration [mol/m^3]
        eclc = 1e-5  # thickness cathode catalyst layer [m]
        gammac = 150  # rugosity cathode [m^2/m^2]
        k1o = 7.068e2  # Kinetic constant [m^7/mol^2/s]
        AH2O2 = 42450  # Activation energy [J/mol]
        alfa = 0.5  # Trasfers coefficient of the reaction [-]
        cH = (1980 + 32.4 * self.lambdam) / ((1 + 0.0648 * self.lambdam) * EW)  # Concentration of H+
        k1 = k1o * np.exp(-AH2O2 / (self.R * Tk)) * np.exp(-alfa * self.F * eta / (self.R * self.Tref))  # Kinetic constant [m^7/mol^2/s]
        R1 = k1 * cO2 * cH ** 2  # Kinetic rate [mol/m^2/s]
        v1 = gammac * R1 / eclc  # Formation rate [mol/m^3/s] of reaction
        k2 = 1.2e-7  # Kinetic constant [s^(-1)]
        k6 = 2.7e4  # Kinetic constant [m^3/mol/s]
        k7 = 1.2e7  # Kinetic constant [m^3/mol/s]
        k10 = 1e3  # Kinetic constant [m^3/mol/s]
        e = k7 * cO2 + k10 * Cmemb - vH2O / eclc
        A2 = -3 * k2 + vH2O / eclc
        B = e * vH2O / (eclc * k6) - v1 - e * k2 / k6
        C = -e * v1 / k6
        CH2O2 = (-B + np.sqrt(B ** 2 - 4 * A2 * C)) / (2 * A2)  # Hydrogen peroxide concetration [mol/m^3]
        CHO = vH2O / (eclc * k6) - k2 / k6 - v1 / (k6 * CH2O2)  # Hydroxil concentration [mol/m^3]
        return CH2O2, CHO

    def FRRlm(self, CHO, t):
        ############### DEPRECATED ################
        print("This function is deprecated, use FRRlm_mod instead")
        k10 = 1e3  # Kinetic constant [m^3/mol/s]
        EW = 1.100  # Nafion equivalent weight [kg/mol]
        rhonaf = 1980  # Nafion dry membrane density [kg/m^3]
        Cmemb = rhonaf / EW  # Membrane concentration [mol/m^3]
        v10 = k10 * CHO * Cmemb  # Chemical reaction rate [mol/m^3/s]
        vF = 3.6 * v10  # F- Formation rate [mol/m^3/s]
        MMF = 18.998403  # Molar mass of the fluoride ions [g/mol]
        FRR = vF * MMF * (self.lm * 10 ** (-2)) * 3600 / (10 ** 4)  # Fluoride Release Rate [g/cm^2/h]
        # FRR = vF * MMF * (self.lm * 10 ** (-6)) * 3600  # Fluoride Release Rate [g/cm^2/h]
        TR = FRR / (0.82 * 2)  # thickness reduction rate [cm/h]
        if TR > 0:
            lm = self.lm - TR * t  # membrane thickness [cm]
        else:
            lm = self.lm
        return FRR, lm
    
    def FRRlm_mod(self, CHO, dt):
        k10 = 1e3  # Kinetic constant [m^3/mol/s]
        EW = 1.100  # Nafion equivalent weight [kg/mol]
        rhonaf = 1980  # Nafion dry membrane density [kg/m^3]
        Cmemb = rhonaf / EW  # Membrane concentration [mol/m^3]
        v10 = k10 * CHO * Cmemb  # Chemical reaction rate [mol/m^3/s]
        vF = 3.6 * v10  # F- Formation rate [mol/m^3/s]
        MMF = 18.998403  # Molar mass of the fluoride ions [g/mol]
        FRR = vF * MMF * (self.lm * 10 ** (-2)) * 3600 / (10 ** 4)  # Fluoride Release Rate [g/cm^2/h]
        TR = FRR / (0.82 * 2)  # thickness reduction rate [cm/h]
        # Update membrane thickness 
        self.lm -= TR * dt
        # self.lm = self.initial_thickness - TR * t
        if self.lm < 0:
            self.lm = 0  # Ensure membrane thickness doesn't go negative
        return FRR, self.lm    

    def Vact_deg(self, Tk, i_cell, pres, lm, exact):
        EW = 1.100  # Nafion equivalent weight [kg/mol]
        w = self.lambdam * self.MmH2O * 10 ** (-3) / EW  # Normalized water uptake
        SH2 = 1 / (w * 1.09e5 * np.exp(77 / Tk)) * 10 ** 5  # Hydrogen solubility in water [mol/m^3/bar]
        DH2 = 1.23e-6 * np.exp(-(-2602) / Tk)  # Hydrogen diffusivity in water [m^2/s]
        ppH2 = pres - self.PsatH2O(Tk)  # Hydrogen partial pressure [bar]
        iH2 = SH2 * DH2 * ppH2 / lm * 2 * self.F  # gas crossover current [A/cm^2]
        i0_an = self.roughness_an * self.i0ref_an * np.exp(-(self.Ea_an / self.R) * (1 / Tk - 1 / 353))
        i0_cat = self.roughness_cat * self.i0ref_cat * np.exp(-(self.Ea_cat / self.R) * (1 / Tk - 1 / 353))
        # c = self.R * Tk / (2 * self.Alpha_an * self.F)
        c_an = self.R * Tk / (self.Alpha_an * self.F)
        c_cat = self.R * Tk / (self.Alpha_cat * self.F)

        # nx = self.R * Tk * np.log(iH2 / i0_cat) / self.F  # gas crossover overvoltage [V]
        if exact:
            b1 = np.arcsinh(i_cell / (2 * i0_an))
            b2 = np.arcsinh(i_cell / (2 * i0_cat))
        else:
            if i_cell == 0:
                b1 = 0
                b2 = 0
            else:
                b1 = np.log(i_cell / i0_an)
                b2 = np.log(i_cell / i0_cat)
        ## TODO: Include again nx term (it has been omited because of its dependence to the membrane thickness to simply the final equation)
        # Vact_deg = c_an * b1 + c_cat * b2 + nx  # [V]
        Vact_deg = c_an * b1 + c_cat * b2  # [V]

        return Vact_deg

    def VOhm_deg(self, Tk, i_cell, lm):
        ## TODO: Modify the conductivity according to Chandesris
        # Area Specific ohmic Resistance
        ## TODO: Check why in the original equations from the thesis this equation changes
        # r = lm * 1 / ((0.005139 * self.lambdam + 0.00326) * np.exp(1268 * (1 / 303 - 1 / Tk)))  # [Ohm*cm^2]
        sigma = ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))
        # r = lm * 1 / ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))  # [Ohm*cm^2]
        # Without degradation
        # r = lm/sigma

        # Include degradation in the Ohm voltage
        sigma = ((0.005139 * self.lambdam - 0.00326) * np.exp(1267 * (1 / 303 - 1 / Tk)))
        sigma_deg = ((self.lm/self.initial_thickness)**2) * sigma
        r = self.lm/sigma_deg
        # r = r *((self.lm/lm)**2)  
        VOhm_deg = (i_cell * r)  # [V]
        return VOhm_deg