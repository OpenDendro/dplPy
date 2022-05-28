### Author: Sarah 
### Project: OpenDendro- CSV Format 
### Description: Assigning variables to each value in the tucson formated datasets


with open("test1.txt", "r" ) as rings:
    data= rings.read()
    lines = data.split("\n")
    print (lines[0-1])
    #read every single line
    csvs = []
    for  rows  in lines:
        csvs.append(rows)
    
    #remove empty lines from the file.
    while "" in csvs:
        csvs.remove("")

    allvals = []
    csvtemp = csvs[0].split(",")
    startyear = csvtemp[0]

    for csv in csvs:
        temp = csv.split(",")
        for i in range(1, len(temp)):
            allvals.append(temp[i])

    #convert every element in each list to string- it is easier to manupilate the elements

    