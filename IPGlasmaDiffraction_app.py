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

    predIncoh = np.exp(pred[0, 0:len(t_data_incoherent)])
    predCoh = np.exp(pred[0, len(t_data_incoherent):])

    # make plot
    fig = plt.figure()
    plt.errorbar(t_data_coherent, dsigmadt_data_coherent,
                 dsigmadt_data_coherent_error, color='k', marker='o',
                 linestyle='')
    plt.errorbar(t_data_incoherent, dsigmadt_data_incoherent,
                 dsigmadt_data_incoherent_error, color='k', marker='s',
                 linestyle='')
    plt.plot(t_data_incoherent, predIncoh, '-')
    plt.plot(t_data_coherent, predCoh, '--')
    plt.yscale('log')
    plt.xlim([0, 2.5])
    plt.ylim([1, 5e2])
    plt.xlabel(r"$t$ (GeV$^2$)")
    plt.ylabel(r"$d\sigma/dt$")

    st.pyplot(fig)


if __name__ == '__main__':
    emu = loadEmulator()
    main(emu)
