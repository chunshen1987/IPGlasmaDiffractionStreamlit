import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

from os import path
import sys
sys.path.insert(0, path.abspath('../'))


def parse_model_parameter_file(parfile):
    pardict = {}
    f = open(parfile, 'r')
    for line in f:
        par = line.split("#")[0]
        if par != "":
            par = par.split(":")
            key = par[0]
            val = [ival.strip() for ival in par[1].split(",")]
            for i in range(1, 4):
                val[i] = float(val[i])
            pardict.update({key: val})
    return pardict


@st.cache
def loadEmulator():
    emu = joblib.load("Emulator.joblib")
    return emu


def main(emu):
    # define experimental data
    # the incoherent H1 data  (<W>=75 GeV)
    t_data_incoherent = np.array([0.1, 0.29, 0.52, 0.78, 1.12, 1.55, 2.21])
    dsigmadt_data_incoherent = np.array(
            [47.3, 43.8, 36.7, 27.8, 16.8, 10.05, 6.04])
    dsigmadt_data_incoherent_error = np.array(
            [6.7, 6.0, 5.1, 4.2, 2.59, 1.56, 0.68])

    # input the coherent H1 data  (<W>=75 GeV)
    t_data_coherent = np.array([0.02, 0.08, 0.14, 0.21, 0.3, 0.41, 0.58, 0.9])
    dsigmadt_data_coherent = np.array(
            [336.0, 240.5, 161.2, 111.4, 70.4, 41.2, 18.0, 4.83])
    dsigmadt_data_coherent_error = np.array(
            [18.4, 12.9, 9.3, 7.0, 5.1, 3.7, 2.74, 1.75])

    # The title of the page
    st.title('IPGlasma + Diffraction')

    st.write("This is an interactive web page that emulate "
             + "the J/$$\psi$$ diffractive observables from the IPGlasma model.")
    st.write("This work is based on arXiv:xxxx.xxxxx")
    st.write("One can adjust the model parameters on the left sidebar.")
    st.write("The colored bands in the figure show the emulator estimations "
             + "with their uncertainties. "
             + "The compared experimental data are from the H1 Collaboration, "
             + "[Eur. Phys. J. C 73 (2013) No. 6, 2466]"
             + "(https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-013-2466-y)")

    # Define model parameters in the sidebar
    modelParamFile = "IPGlasmaDiffraction.txt"
    paraDict = parse_model_parameter_file(modelParamFile)
    st.sidebar.header('Model Parameters:')
    params = []     # record the model parameter values
    for ikey in paraDict.keys():
        parMin = paraDict[ikey][1]
        parMax = paraDict[ikey][2]
        parVal = st.sidebar.slider(label=paraDict[ikey][0],
                                   min_value=parMin, max_value=parMax,
                                   value=paraDict[ikey][3],
                                   step=(parMax - parMin)/1000.,
                                   format='%f')
        params.append(parVal)
    params = np.array([params,])

    # make model prediction using the emulator
    pred, predCov = emu.predict(params, return_cov=True)
    pred = np.exp(pred[0, :])
    predErr = np.sqrt(np.diagonal(predCov[0, :, :]))*pred

    predIncoh = pred[0:len(t_data_incoherent)]
    predIncohErr = predErr[0:len(t_data_incoherent)]
    predCoh = pred[len(t_data_incoherent):]
    predCohErr = predErr[len(t_data_incoherent):]

    # make plot
    fig = plt.figure()
    plt.errorbar(t_data_coherent, dsigmadt_data_coherent,
                 dsigmadt_data_coherent_error, color='k', marker='o',
                 linestyle='', label="Coherent, H1")
    plt.errorbar(t_data_incoherent, dsigmadt_data_incoherent,
                 dsigmadt_data_incoherent_error, color='k', marker='s',
                 linestyle='', label="Incoherent, H1")
    plt.fill_between(t_data_incoherent, predIncoh - predIncohErr,
                     predIncoh + predIncohErr, alpha=0.5)
    plt.fill_between(t_data_coherent, predCoh - predCohErr,
                     predCoh + predCohErr, alpha=0.5)
    plt.legend()
    plt.yscale('log')
    plt.xlim([0, 2.5])
    plt.ylim([1, 5e2])
    plt.xlabel(r"$t$ (GeV$^2$)")
    plt.ylabel(r"$d\sigma/dt$ (nb/GeV$^{2}$)")
    plt.text(1.8, 80, r"$W = 75$ GeV")

    st.pyplot(fig)


if __name__ == '__main__':
    emu = loadEmulator()
    main(emu)
