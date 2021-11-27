try:
    with open("/workspaces/dplPy/tests/data/csv/ca533.csv") as rings:
        if rings.endswith('.csv'):
            data = rings.read()
            lines = data.split("\n")
            print (lines[0-1])
        #reads every single line
        csvs = []
        for rows in lines:
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
            #end
    # Echo error if file is corrupt  
except Exception as e:
            print(e)