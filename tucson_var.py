### Author: Anushka 
### Project: OpenDendro- Tucson Format 
### Description: Assigning variables to each value in the tucson formated datasets


with open("test1.txt", "r" ) as rings:
    data= rings.read()
    lines = data.split("\n")
    print (lines[0-2])
    #read every single line
    for  rows  in lines:
        print (rows)
    
    #remove empty lines from the file.

    #convert every element in each list to string- it is easier to manupilate the elements

    for items in lines:
        str(items)  
    #Location 

    ##State/province
    state_province = lines[1,3]

    ##country 
    country= lines[0,2]

    ##Using a dictornary to check if the the place exist or not.  
        # If it does exist then it the program will continue running
        # If not then will tell user to change the location or stop the program completely 
    

    #species
    species = lines[1,4]

    #species code
    species_code= lines[1,5]

    #start year- start of collection
    start_year = int(lines[1,6])

    #end year- completion year
    end_year = int(lines[1,7])

    #latitude, longitude, elevation 
    latitude = lines[2,4]
    longitude = lines[2,5]

    #lead investigater
    lead_investigator= lines[2,2] 

    #site_id
        #Via indexing the first three digits of the site id are assigned to var named site_code and the rest to information.
        #Check if all the site codes are the same for the whole dataset 
    for site_code in lines:
        current=0
        information=0 
        site_id=0 
        while current >=0 and current<=len(lines)-1:
            if current <=3:   
                site_code = lines[current,0,0]       
            if current>=4 and current<=len(lines)-1:
                if lines[current,0,0] == lines[4,0,0]:
                    site_id = lines[current,0,0]
                    information = site_id[3::]  
            site_id= site_code + information 
            #data
            data=[]
            y=1
            past= y-1
            end=[current,-1]
            if site_id == lines[current,0] and start_year <= int(lines[current,1]) and end_year>= int(lines[current,1]):
                for points in lines[current,2:]:
                    pts=[]
                    pts.append(points)
                    # End of data collection for that year
                    if points== "999" or points== "-9999":
                        pts[:-1]
                        data.append(pts)
                        pts=0
            current += 1
            pts=0
   
