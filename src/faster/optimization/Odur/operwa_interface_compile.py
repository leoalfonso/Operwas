import os
import sys

import tkinter as tk
import shapefile as shp
from tkinter import scrolledtext as tkst
from PIL import Image, ImageTk

import csv
from operwas.optimization_library import *

# color codes
IHE_color = '#09aef6'
orange_contrast = '#f65109'
black = '#000000'
white = '#ffffff'
grey = '#696969'
light_grey = '#DCDCDC'

# # main formats

main_text_size = 14

# Logo
path_to_here = os.path.abspath(os.path.dirname(__file__))
logo_file = os.path.join(path_to_here, 'global_part_resized.jpg')
icon_file = os.path.join(path_to_here, 'operwas.ico')

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self,text_area):
        self.text = text_area

class StdoutRedirector(IORedirector):
    '''A class for redirecting stdout to this Text widget.'''
    def write(self,str):
        self.text.insert(tk.INSERT, str)
        self.text.see(tk.END)
        self.text.update()

def check_input_icon(dir_path_with_file_name):

    if os.path.isfile(dir_path_with_file_name):
        check_result = ("✅")
    else:
        check_result = ("❌")

    return check_result

def plot_maps():

    polygons = shp.Reader(path_subcatchments)
    wwtp = shp.Reader(path_ordered_outlets)
    # buildings = shp.Reader(path_buildings)

    plt.figure()

    # for shape in buildings.shapeRecords():
    #     for i in range(len(shape.shape.parts)):
    #         i_start = shape.shape.parts[i]
    #         if i == len(shape.shape.parts) - 1:
    #             i_end = len(shape.shape.points)
    #         else:
    #             i_end = shape.shape.parts[i + 1]
    #         added_legend1 = False
    #         x = [i[0] for i in shape.shape.points[i_start:i_end]]
    #         y = [i[1] for i in shape.shape.points[i_start:i_end]]
    #         if not added_legend1:
    #             plt.plot(x, y, color='orange', linestyle='dashed', label='Population distribution', linewidth=1)
    #             added_legend1 = True
    #         else:
    #             plt.plot(x, y, color='grey', linestyle='dashed', linewidth=1)


    added_legend = False
    for shape in polygons.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        if not added_legend:
            plt.plot(x, y, color='grey', linestyle='dashed', label='Catchments', linewidth=1)
            added_legend = True
        else:
            plt.plot(x, y, color='grey', linestyle='dashed', linewidth=1)

    types_treatment = ['cas with agr. reuse', 'mbr with urb. reuse', 'cas with agr. reuse', 'mbr with urb. reuse', 'cas with agr. reuse', 'mbr with urb. reuse']

    type2legend = {
        'cas with agr. reuse': ('green', 'p', 'Wastewater treatment plant (cas)'),
        'mbr with urb. reuse': ('red', '^', 'Wastewater treatment plant (mbr)')
    }

    added_legend_per_type = {'cas with agr. reuse': False, 'mbr with urb. reuse': False}
    for shape, type in zip(wwtp.shapeRecords(), types_treatment):
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]

        # Get color maker and label for this type
        color, marker, label = type2legend[type]
        if added_legend_per_type[type]:
            label = None
        else:
            added_legend_per_type[type] = True
        plt.plot(x, y, color=color, marker=marker, markersize=5, label=label)

    plt.legend(loc='lower left')
    plt.show()


def main():
    # INTERFACE

    root = tk.Tk()

    # text on the side
    var = tk.StringVar()
    label = tk.Label( root, textvariable=var)
    var.set("Progress window:")
    label.config(font=("Calibri", 14, 'bold'), fg=black)
    label.place(relx=0.04, rely=0.7)


    text = tkst.ScrolledText(root, wrap=tk.WORD, height=10, width=65)
    text.place(relx=0.04, rely=0.75)

    sys.stdout = StdoutRedirector(text)

    side_allign = 0.52

    # size of the window
    root.geometry('1100x800')

    # logo no topo
    root.wm_iconbitmap(icon_file)
    root.wm_title("Operwas | Optimization, exploration and research of wastewater treatment locations")

    # internal logo
    load = Image.open(logo_file)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render)
    img.image = render
    img.place(relx=side_allign, rely=0)

    # text on the side
    var = tk.StringVar()
    label = tk.Label( root, textvariable=var)
    var.set("Oper")
    label.config(font=("Calibri", 40, 'bold'), fg=IHE_color)
    label.place(relx=0.04, rely=0.02)

    half = tk.StringVar()
    halflabel = tk.Label( root, textvariable=half)
    half.set("w")
    halflabel.config(font=("Calibri", 40, 'bold'), fg=orange_contrast)
    halflabel.place(relx=0.143, rely=0.02)

    end = tk.StringVar()
    endlabel = tk.Label( root, textvariable=end)
    end.set("as")
    endlabel.config(font=("Calibri", 40, 'bold'), fg=IHE_color)
    endlabel.place(relx=0.18, rely=0.02)

    # text on the side
    var = tk.StringVar()
    label = tk.Label( root, textvariable=var)
    var.set("Optimization, exploration and research of wastewater treatment locations")
    label.config(font=("Calibri", 20, 'bold'), wraplength=600, justify='left', fg=grey)
    label.place(relx=0.04, rely=0.10)

    # window with checks
    checkframe = tk.Frame(root, bg=white, bd=5, height=350, width= 500)
    checkframe.place(relx=0.04, rely=0.25)

    # input data title (inside of main window)
    var = tk.StringVar()
    label = tk.Label(checkframe, bg=white, textvariable=var)
    var.set("Check your input files: ")
    label.config(font=("Calibri", main_text_size, 'bold'))
    label.place(relx=0, rely=0)

    # Checks

    rely_start = 0.10
    distance = 0.17
    wraplength = 450

    # Get root of data to be checked
    root_dir = os.path.expanduser(os.path.join("~", ".operwas_data"))
    inputs_dir = os.path.join(root_dir, 'inputs')

    ## Paths that should be in the input folder:
    dem_dir = os.path.join(inputs_dir, 'DEM_filled.tif')
    channels_dir = os.path.join(inputs_dir, 'channels.shp')
    grid_pop_data_dir = os.path.join(inputs_dir, 'grid_data.shp')
    excel_dir = os.path.join(inputs_dir, 'operwas_user_input.xlsx')
    wwtp_list_dir = os.path.join(inputs_dir, 'wwtp_locations.csv')

    ## Assign the right emoticon
    check_dem = check_input_icon(dem_dir)
    check_channels = check_input_icon(channels_dir)
    check_grid = check_input_icon(grid_pop_data_dir)
    check_excel = check_input_icon(excel_dir)
    check_wwtp_list = check_input_icon(wwtp_list_dir)

    # Display checks
    ConfirmDem = tk.Label(checkframe, text=f"{check_dem}  'DEM.tif' is in folder .operwas_data/inputs")
    ConfirmDem.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=wraplength, )
    ConfirmDem.place(relx=0, rely=rely_start)

    ConfirmChannel = tk.Label(checkframe, text=f"{check_channels}  'channels.shp' is in folder .operwas_data/inputs")
    ConfirmChannel.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=wraplength, )
    ConfirmChannel.place(relx=0, rely=rely_start + distance)

    ConfirmGrid = tk.Label(checkframe, text=f"{check_grid}  'grid_data.shp' is in folder .operwas_data/inputs")
    ConfirmGrid.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=wraplength, )
    ConfirmGrid.place(relx=0, rely=rely_start + 2*distance)

    ConfirmExcel = tk.Label(checkframe, text=f"{check_excel}  'operwas_user_input.xlsx' is in folder .operwas_data/inputs")
    ConfirmExcel.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=wraplength, )
    ConfirmExcel.place(relx=0, rely=rely_start + 3 * distance)

    ConfirmWWTPList = tk.Label(checkframe, text=f"{check_wwtp_list}  'wwtp_locations.csv' is in folder .operwas_data/inputs")
    ConfirmWWTPList.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=wraplength, )
    ConfirmWWTPList.place(relx=0, rely=rely_start + 4 * distance)


    #checks = [check_root, check_grid, check_dem, check_channel, check_input]



    # main window
    mainframe = tk.Frame(root, bg=white, bd=5, height=280, width= 500)
    mainframe.place(relx=side_allign, rely=0.25)

    # input data title (inside of main window)
    var = tk.StringVar()
    label = tk.Label(mainframe, bg=white, textvariable=var)
    var.set("Optimise WWTP location based on a given number of WWTPs")
    label.config(font=("Calibri", main_text_size, 'bold'))
    label.place(relx=0, rely=0)

    # first entry of data (inside of main window)
    L1 = tk.Label(mainframe, text="Number of WWTPs to locate:")
    L1.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=400,)
    L1.place(relx=0.02, rely=0.15)
    E1 = tk.Entry(mainframe, bd =2)
    E1.place(relx=0.7, rely=0.15)

    # second entry of data (inside of main window)
    L2 = tk.Label(mainframe, text="Number of options to display:")
    L2.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=300,)
    L2.place(relx=0.02, rely=0.25)
    E2 = tk.Entry(mainframe, bd =2)
    E2.place(relx=0.7, rely=0.25)

    # run button 1 (opera, fixed number of function evaluations (uses both user entries), and runs NSGA2
    run_button1 = tk.Button(mainframe, text ="Optimal WWTPs locations", font=("Calibri", main_text_size),
                           command=lambda:Operwa(E1.get(), E2.get()))
    run_button1.place(relx=0.35, rely=0.35)

# Second run option

    var = tk.StringVar()
    label = tk.Label(mainframe, bg=white, textvariable=var)
    var.set("Optimise number and location of WWTPs")
    label.config(font=("Calibri", main_text_size, 'bold'))
    label.place(relx=0, rely=0.6)

    # first entry of data (inside of main window)
    L3 = tk.Label(mainframe, text="Number of runs (options):")
    L3.config(font=("Calibri", main_text_size), fg=black, bg=white, wraplength=400,)
    L3.place(relx=0.02, rely=0.75)
    E3 = tk.Entry(mainframe, bd =2)
    E3.place(relx=0.7, rely=0.75)


    # run button 2 (operwas2, decides number of treatment plants, uses only number of runs (second entry)
    run_button2 = tk.Button(mainframe, text ="Number and location of WWTPs", font=("Calibri", main_text_size),
                           command=lambda:Operwas2(E3.get()))
    # run_button = tk.Button(mainframe, text ="Let's run it!", font=("Calibri", 15), command=lambda:[call_back, progress_bar])
    run_button2.place(relx=0.3, rely=0.85)


    # results window
    resultframe = tk.Frame(root, bg=white, bd=5, height=200, width= 500)
    resultframe.place(relx=side_allign, rely=0.65)

    # input data title (inside of main window)
    var = tk.StringVar()
    label = tk.Label(resultframe, bg=white, textvariable=var)
    var.set("Look at the results:")
    label.config(font=("Calibri", main_text_size, 'bold'))
    label.place(relx=0, rely=0)


    def import_partial_results():
        os.startfile(os.path.join(root_dir, 'results', 'partial_results.csv' ))



    def import_total_results():
        os.startfile(os.path.join(root_dir, 'results', 'total_results.csv'))

    def import_optimization_results():
        os.startfile(os.path.join(root_dir,'results','list_of_results.csv'))

    # sheet button
    sheet_button = tk.Button(resultframe, text ="Calculated data for each wastewater treatment plant", wraplength=300,
                             font=("Calibri", main_text_size), command=import_partial_results)
    sheet_button.place(relx=0, rely=0.3)

    sheet_button = tk.Button(resultframe, text ="Calculated data (total)", font=("Calibri", main_text_size), wraplength=150,
                             command=import_total_results)
    sheet_button.place(relx=0.65, rely=0.3)

    # map button
    map_button = tk.Button(resultframe, text ="Sample maps", font=("Calibri", main_text_size), command=lambda:plot_maps())
    map_button.place(relx=0, rely=0.7)

    # optimization solutions buttom
    opt_button = tk.Button(resultframe, text ="Results of the optimization", font=("Calibri", main_text_size), command=import_optimization_results)
    opt_button.place(relx=0.4, rely=0.7)

    #To make the window remain opened
    root.mainloop()


if __name__ == "__main__":
    main()
