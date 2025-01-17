#!/usr/bin/python

#Image querying script written by Tamara Berg,
#and extended heavily James Hays

#9/26/2007 added dynamic timeslices to query more efficiently.
#8/18/2008 added new fields and set maximum time slice.
#8/19/2008 this is a much simpler function which gets ALL geotagged photos of
# sufficient accuracy.  No queries, no negative constraints.
# divides up the query results into multiple files
# 1/5/2009
# now uses date_taken instead of date_upload to get more diverse blocks of images
# 1/13/2009 - uses the original im2gps keywords, not as negative constraints though

import sys, string, math, time, socket
from flickrapi2 import FlickrAPI
from datetime import datetime

socket.setdefaulttimeout(30)  #30 second time out on sockets before they throw
#an exception.  I've been having trouble with urllib.urlopen hanging in the 
#flickr API.  This will show up as exceptions.IOError.

#the time out needs to be pretty long, it seems, because the flickr servers can be slow
#to respond to our big searches.

print sys.argv 
if len(sys.argv) > 1:
    print "Reading queries from file " + sys.argv[1]
    query_file_name = sys.argv[1] #0 is the command name.
else:
    print "No command line arguments, reading queries from " + 'queries.txt'
    query_file_name = 'place_rec_queries.txt'
    #query_file_name = 'place_rec_queries_fall08.txt'

###########################################################################
# Modify this section to reflect your data and specific search 
###########################################################################
# flickr auth information:
# change these to your flickr api keys and secret
flickrAPIKey = "b34798c5e0536c85abda421741c05498"  # API key
flickrSecret = "408462efc5130a7e"                  # shared "secret"

query_file = open(query_file_name, 'r')

#aggregate all of the positive and negative queries together.
pos_queries = []  #an empty list
neg_queries = ''  #a string
num_queries = 0

for line in query_file:
    if line[0] != '#' and len(line) > 2:  #line end character is 2 long?
      #print line[0:len(line)-2]
      if line[0] != '-':
        pos_queries = pos_queries + [line[0:len(line)-2]]
        num_queries = num_queries + 1
      if line[0] == '-':
        neg_queries = neg_queries + ' ' + line[0:len(line)-2]
        
query_file.close()
print 'positive queries:  '
print pos_queries
print 'negative queries:  ' + neg_queries
print 'num_queries = ' + str(num_queries)

# make a new FlickrAPI instance
fapi = FlickrAPI(flickrAPIKey, flickrSecret)

for current_tag in range(0, num_queries):
  
    # change this to the location where you want to put your output file
    out_file = open('./dataset/' + pos_queries[current_tag] + '.txt','w')
    ###########################################################################
    
    #form the query string.
    query_string = pos_queries[current_tag] + neg_queries
    print '\n\nquery_string is ' + query_string
    total_images_queried = 0;


    # number of seconds to skip per query  
    #timeskip = 62899200 #two years
    #timeskip = 604800  #one week
    #timeskip = 172800  #two days
    timeskip = 86400 #one day
    #timeskip = 3600 #one hour
    #timeskip = 2257 #for resuming previous query

    #mintime = 1121832000 #from im2gps
    #mintime = 1167407788 # resume crash england
    #mintime = 1177828976 #resume crash japan
    #mintime = 1187753798 #resume crash greece
    mintime = 1171416400 #resume crash WashingtonDC
    maxtime = mintime+timeskip
    endtime =  1192165200  #10/12/2007, at the end of im2gps queries

    #this is the desired number of photos in each block
    desired_photos = 250

    print datetime.fromtimestamp(mintime)
    print datetime.fromtimestamp(endtime)

    while (maxtime < endtime and total_images_queried < 4000):

        #new approach - adjust maxtime until we get the desired number of images 
        #within a block. We'll need to keep upper bounds and lower
        #lower bound is well defined (mintime), but upper bound is not. We can't 
        #search all the way from endtime.
    
        lower_bound = mintime + 900 #lower bound OF the upper time limit. must be at least 15 minutes or zero results
        upper_bound = mintime + timeskip * 20 #upper bound of the upper time limit
        maxtime     = .95 * lower_bound + .05 * upper_bound
    
        print '\nBinary search on time range upper bound' 
        print 'Lower bound is ' + str(datetime.fromtimestamp(lower_bound))
        print 'Upper bound is ' + str(datetime.fromtimestamp(upper_bound))
    
        keep_going = 6 #search stops after a fixed number of iterations
        while( keep_going > 0 and maxtime < endtime):
        
            try:
                rsp = fapi.photos_search(api_key=flickrAPIKey,
                                        ispublic="1",
                                        media="photos",
                                        per_page="250", 
                                        page="1",
                                        #has_geo = "1", #bbox="-180, -90, 180, 90",
                                        text=query_string,
                                        #accuracy="6", #6 is region level.  most things seem 10 or better.
                                        min_upload_date=str(mintime),
                                        max_upload_date=str(maxtime))
                                        ##min_taken_date=str(datetime.fromtimestamp(mintime)),
                                        ##max_taken_date=str(datetime.fromtimestamp(maxtime)))
                #we want to catch these failures somehow and keep going.
                time.sleep(1)
                fapi.testFailure(rsp)
                total_images = rsp.photos[0]['total'];
                null_test = int(total_images); #want to make sure this won't crash later on for some reason
                null_test = float(total_images);
        
                print '\nnumimgs: ' + total_images
                print 'mintime: ' + str(mintime) + ' maxtime: ' + str(maxtime) + ' timeskip:  ' + str(maxtime - mintime)
            
                if( int(total_images) > desired_photos ):
                    print 'too many photos in block, reducing maxtime'
                    upper_bound = maxtime
                    maxtime = (lower_bound + maxtime) / 2 #midpoint between current value and lower bound.
                
                if( int(total_images) < desired_photos):
                    print 'too few photos in block, increasing maxtime'
                    lower_bound = maxtime
                    maxtime = (upper_bound + maxtime) / 2
                
                print 'Lower bound is ' + str(datetime.fromtimestamp(lower_bound))
                print 'Upper bound is ' + str(datetime.fromtimestamp(upper_bound))
            
                if( int(total_images) > 0): #only if we're not in a degenerate case
                    keep_going = keep_going - 1
                else:
                    upper_bound = upper_bound + timeskip;    
            
            except KeyboardInterrupt:
                print('Keyboard exception while querying for images, exiting\n')
                raise
            except:
                print sys.exc_info()[0]
                #print type(inst)     # the exception instance
                #print inst.args      # arguments stored in .args
                #print inst           # __str__ allows args to printed directly
                print ('Exception encountered while querying for images\n')

        #end of while binary search    
        print 'finished binary search'
        
        s = '\nmintime: ' + str(mintime) + ' maxtime: ' + str(maxtime)
        print s
        out_file.write(s + '\n') 

        i = getattr(rsp,'photos',None)
        if i:
                
            s = 'numimgs: ' + total_images
            print s
            out_file.write(s + '\n')
            
            current_image_num = 1;
            
            num = int(rsp.photos[0]['pages'])
            s =  'total pages: ' + str(num)
            print s
            out_file.write(s + '\n')
            
            #only visit 16 pages max, to try and avoid the dreaded duplicate bug
            #16 pages = 4000 images, should be duplicate safe.  Most interesting pictures will be taken.
            
            num_visit_pages = min(16,num)
            
            s = 'visiting only ' + str(num_visit_pages) + ' pages ( up to ' + str(num_visit_pages * 250) + ' images)'
            print s
            out_file.write(s + '\n')
            
            total_images_queried = total_images_queried + min((num_visit_pages * 250), int(total_images))

            #print 'stopping before page ' + str(int(math.ceil(num/3) + 1)) + '\n'
        
            pagenum = 1;
            while( pagenum <= num_visit_pages ):
            #for pagenum in range(1, num_visit_pages + 1):  #page one is searched twice
                print '  page number ' + str(pagenum)
                try:
                    rsp = fapi.photos_search(api_key=flickrAPIKey,
                                        ispublic="1",
                                        media="photos",
                                        per_page="250", 
                                        page=str(pagenum),
                                        sort="interestingness-desc",
                                        #has_geo = "1", #bbox="-180, -90, 180, 90",
                                        text=query_string,
                                        #accuracy="6", #6 is region level.  most things seem 10 or better.
                                        extras = "tags, original_format, license, geo, date_taken, date_upload, o_dims, views",
                                        min_upload_date=str(mintime),
                                        max_upload_date=str(maxtime))
                                        ##min_taken_date=str(datetime.fromtimestamp(mintime)),
                                        ##max_taken_date=str(datetime.fromtimestamp(maxtime)))
                    time.sleep(1)
                    fapi.testFailure(rsp)
                except KeyboardInterrupt:
                    print('Keyboard exception while querying for images, exiting\n')
                    raise
                except:
                    print sys.exc_info()[0]
                    #print type(inst)     # the exception instance
                    #print inst.args      # arguments stored in .args
                    #print inst           # __str__ allows args to printed directly
                    print ('Exception encountered while querying for images\n')
                else:

                    # and print them
                    k = getattr(rsp,'photos',None)
                    if k:
                        m = getattr(rsp.photos[0],'photo',None)
                        if m:
                            for b in rsp.photos[0].photo:
                                if b!=None:
                                    out_file.write('photo: ' + b['id'] + ' ' + b['secret'] + ' ' + b['server'] + '\n')
                                    out_file.write('owner: ' + b['owner'] + '\n') 
                                    out_file.write('title: ' + b['title'].encode("ascii","replace") + '\n')
                                    
                                    out_file.write('originalsecret: ' + b['originalsecret'] + '\n')
                                    out_file.write('originalformat: ' + b['originalformat'] + '\n')
                                    out_file.write('o_height: ' + b['o_height'] + '\n')
                                    out_file.write('o_width: ' + b['o_width'] + '\n')
                                    out_file.write('datetaken: ' + b['datetaken'].encode("ascii","replace") + '\n')
                                    out_file.write('dateupload: ' + b['dateupload'].encode("ascii","replace") + '\n')
                                    
                                    out_file.write('tags: ' + b['tags'].encode("ascii","replace") + '\n')
                                    
                                    out_file.write('license: ' + b['license'].encode("ascii","replace") + '\n')
                                    out_file.write('latitude: '  + b['latitude'].encode("ascii","replace") + '\n')
                                    out_file.write('longitude: ' + b['longitude'].encode("ascii","replace") + '\n')
                                    out_file.write('accuracy: '  + b['accuracy'].encode("ascii","replace") + '\n')
                                    
                                    out_file.write('views: ' + b['views'] + '\n')
                                    out_file.write('interestingness: ' + str(current_image_num) + ' out of ' + str(total_images) + '\n');
                                    out_file.write('\n')
                                    current_image_num = current_image_num + 1;
                    pagenum = pagenum + 1;  #this is in the else exception block.  It won't increment for a failure.

            #this block is indented such that it will only run if there are no exceptions
            #in the original query.  That means if there are exceptions, mintime won't be incremented
            #and it will try again
            timeskip = maxtime - mintime #used for initializing next binary search
            mintime  = maxtime

    out_file.write('Total images queried: ' + str(total_images_queried) + '\n')
    out_file.close
