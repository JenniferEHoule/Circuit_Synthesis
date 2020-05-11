""" plots.py

These are loosely based on the final figure in ex2_Y.m from the Matrix Fitting Toolbox [1]. These figures are
similar to each other.

Author: Jennifer Houle
Date: 4/6/2020

B. Gustavsen, Matrix Fitting Toolbox, The Vector Fitting Website. March 20, 2013. Accessed on:
Feb. 25, 2020. [Online]. Available:
https://www.sintef.no/projectweb/vectorfitting/downloads/matrix-fitting-toolbox/.

"""

from math import pi, sqrt
import numpy as np

from matplotlib import pyplot as plt


def plot_figure_11(s, bigYfit, bigYfit_passive, SER):
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = 'dotted'

    fig = plt.figure(11, figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1)
    freq = (s / (2 * np.pi * 1j)).real
    Ns = freq.shape[0]

    Nc = SER['D'].shape[0]
    for row in range(Nc):
        for col in range(Nc):
            dum1 = bigYfit[row, col, :]
            dum2 = bigYfit_passive[row, col, :]
            ax.plot(freq, np.abs(dum1), color='b', linewidth=1, label='Original')
            ax.plot(freq, np.abs(dum2), color='r', linewidth=1, label='Perturbed')
            diff = dum2 - dum1
            ax.plot(freq, np.abs(diff), color='g', linewidth=1, label='Difference')

            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Admittance')
            plt.title("Model Comparison")
            plt.legend()
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'model_comparison_{row},{col}')
            print(f"RMS error Y{row+1}{col+1}: {np.sqrt(np.sum(np.abs(diff ** 2))) / np.sqrt(Ns)}")
    plt.show()
    return

def plot_figure_11_individual(s, bigYfit, bigYfit_passive, Nc, name1, name2, example_name):
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = 'dotted'

    freq = (s / (2 * np.pi * 1j)).real

    Ns = freq.shape[0]

    for row in range(Nc):
        for col in range(Nc):
            fig = plt.figure(figsize=(8, 7))
            ax = fig.gca()
            dum1 = bigYfit[row, col, :]
            dum2 = bigYfit_passive[row, col, :]
            ax.plot(freq, np.abs(dum1), color='b', linewidth=1, label=name1)
            ax.plot(freq, np.abs(dum2), color='r', linewidth=1, label=name2)
            diff = dum2 - dum1
            ax.plot(freq, np.abs(diff), color='g', linewidth=1, label='Difference')

            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Admittance')
            plt.title(f"Model Comparison: Y{row+1}{col+1}")
            plt.legend()
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'model_comparison_magnitude_{name1}_{name2}_Y{row+1}{col+1}_{example_name}')
            print(f"RMS error of magnitude, Y{row+1}{col+1}: {np.sqrt(np.sum(np.abs(diff ** 2))) / np.sqrt(Ns)}")


            dum1 = bigYfit[row, col, :]
            dum2 = bigYfit_passive[row, col, :]
            ax.plot(freq, 180 / pi * np.unwrap(np.angle(dum1)), color='b', linewidth=1, label=name1)
            ax.plot(freq, 180 / pi * np.unwrap(np.angle(dum2)), color='r', linewidth=1, label=name2)
            diff = 180 / pi * np.unwrap(np.angle(dum2)) - 180 / pi * np.unwrap(np.angle(dum1))
            ax.plot(freq, np.abs(diff), color='g', linewidth=1, label='Difference')

            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase Angle [deg]')
            plt.title(f"Model Comparison: Y{row+1}{col+1}")
            plt.legend()
            plt.yscale('linear')
            plt.tight_layout()
            plt.savefig(f'model_comparison_phase_{name1}_{name2}_Y{row+1}{col+1}_{example_name}')
            rmserr = np.sqrt(np.sum(np.abs((np.angle(dum1) - np.angle(dum2)) ** 2))) / np.sqrt(Ns)
            print(f"RMS error of phase, Y{row+1}{col+1}: {rmserr}")

    return

def plot_figure_11_mag_phase(s, bigYfit, bigYfit_passive, Nc, name1, name2, example_name):
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = 'dotted'

    freq = (s / (2 * np.pi * 1j)).real

    Ns = freq.shape[0]

    for row in range(Nc):
        for col in range(Nc):
            fig = plt.figure(figsize=(8, 7))
            dum1 = bigYfit[row, col, :]
            dum2 = bigYfit_passive[row, col, :]

            print(f"RMS Error Y{row + 1}{col + 1} (Mag): {np.sqrt(np.sum(np.abs((np.absolute(dum1) - np.absolute(dum2)) ** 2))) / np.sqrt(Ns)}")
            print(f"RMS Error Y{row + 1}{col + 1} (Phase): {np.sqrt(np.sum(np.abs((np.angle(dum1) - np.angle(dum2)) ** 2))) / np.sqrt(Ns)}")

            plt.subplot(121)
            plt.title(f"Y{row + 1}{col + 1}, Magnitude")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.plot(freq, np.absolute(dum2), label=name2, color='k')
            plt.plot(freq, np.absolute(dum1), label=name1, color='g')
            plt.plot(freq, np.absolute(dum1 - dum2), label='Difference', color='r')
            plt.yscale('log')
            plt.legend()

            plt.subplot(122)
            plt.title(f"Y{row + 1}{col + 1}, Phase")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Phase")
            plt.plot(freq, np.angle(dum1) * 180 / np.pi, label=name1, color='k')
            plt.plot(freq, np.angle(dum2) * 180 / np.pi, label=name2, color='g')
            # plt.plot(freq, np.angle(dum1 - dum2) * 180 / np.pi, label='Difference', color='r')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'model_comparison_mag-phase_{name1}_{name2}_Y{row+1}{col+1}_{example_name}')

    return

def plot_figure_mag_phase_from_ADS(freq, bigYfit, bigYfit_passive, Nc, name1, name2, example_name):
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = 'dotted'

    # freq = (s / (2 * np.pi * 1j)).real

    Ns = freq.shape[0]

    for row in range(Nc):
        for col in range(Nc):
            fig = plt.figure(figsize=(8, 7))
            ax = fig.gca()
            dum1 = bigYfit[row, col, :]
            dum2 = bigYfit_passive[row, col, :]

            print(f"RMS Error Y{row + 1}{col + 1} (Mag): {np.sqrt(np.sum(np.abs((np.absolute(dum1) - np.absolute(dum2)) ** 2))) / np.sqrt(Ns)}")
            print(f"RMS Error Y{row + 1}{col + 1} (Phase): {np.sqrt(np.sum(np.abs((np.angle(dum1) - np.angle(dum2)) ** 2))) / np.sqrt(Ns)}")

            plt.subplot(121)
            plt.title(f"Y{row + 1}{col + 1}, Magnitude")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Magnitude")
            plt.plot(freq, np.absolute(dum2), label=name2)
            plt.plot(freq, np.absolute(dum1) - np.absolute(dum2), label='Difference')
            plt.plot(freq, np.absolute(dum1), label=name1)
            plt.yscale('log')
            plt.legend()

            plt.subplot(122)
            plt.title(f"Y{row + 1}{col + 1}, Phase")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Phase")
            plt.plot(freq, np.angle(dum1) * 180 / np.pi, label=name1)
            plt.plot(freq, np.angle(dum2) * 180 / np.pi, label=name2)
            plt.plot(freq, np.angle(dum1) * 180 / np.pi - np.angle(dum2) * 180 / np.pi, label='Difference')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'model_comparison_mag-phase_{name1}_{name2}_Y{row+1}{col+1}_{example_name}')

            # plt.show()

    return

def plot_magnitude_phase_VF_format(s, f_original, final_f, Nc, Ns, name1, name2, example_name):
    number_of_ports = sqrt(final_f.shape[0])
    for number in range(f_original.shape[0]):
        port1 = int(1 + number // number_of_ports)
        port2 = int(number % number_of_ports + 1)
        print(f"RMS Error Y{port1}{port2} (Mag): {np.sqrt(np.sum(np.abs((np.absolute(f_original[number, :]) - np.absolute(final_f[number, :])) ** 2))) / np.sqrt(Nc * Ns)}")
        print(f"RMS Error Y{port1}{port2} (Phase): {np.sqrt(np.sum(np.abs((np.angle(f_original[number, :]) - np.angle(final_f[number, :])) ** 2))) / np.sqrt(Nc * Ns)}")

        plt.subplot(121)
        plt.title(f"Y{port1}{port2}, Magnitude")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.plot(s, np.absolute(f_original[number, :]), label=name1)
        plt.plot(s, np.absolute(final_f[number, :]), label=name2)
        plt.plot(s, np.absolute(final_f[number, :]) - np.absolute((f_original[number, :])), label='difference')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()

        plt.subplot(122)
        plt.title(f"Y{port1}{port2}, Phase")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase")
        plt.plot(s, np.angle(f_original[number, :]) * 180 / np.pi, label=name1)
        plt.plot(s, np.angle(final_f[number, :]) * 180 / np.pi, label=name2)
        plt.xscale('log')
        plt.legend()
        plt.savefig(f'data_comparison_{example_name}_Y{port1}{port2}')
        plt.tight_layout()
        plt.show()
