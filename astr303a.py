import numpy as np
import pylab as py
from scipy import optimize
from matplotlib.patches import Ellipse
import subprocess
import sys
import re
import os
import io
import scipy
import shutil
from matplotlib import pylab
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Create a pdf file that all figures will be saved to.
global pp
pp = PdfPages('GALAXIES.pdf')

#filename of text file containing SDSS-DR13 data.
filename = 'specobjdataset.txt'

#read data into memory and organize into an array
def readData(filename):
    #read data set into memory
    with open(filename,'r') as file:
        data = file.readlines()
        file.close()

    #parse data into array of galaxies with rows:
    #[objID(0),ra(1),dec(2),mag(3),magErr(4),ext(5),z(6),zErr(7)]
    dataarray = []
    #cycle through all lines of text in specobjdataset.txt
    for i in range(1,len(data)):
        line = data[i].replace('\n','') #remove white space and empty lines
        line = line.split(',') #split lines where commas are found
        objID = line[6] 
        ra = float(line[7])
        dec = float(line[2])
        mag = float(line[4])
        magErr = float(line[5])
        ext = float(line[3])
        z = float(line [0])
        zErr = float(line[1])
        dataarray.append([objID,ra,dec,mag,magErr,ext,z,zErr]) #append galaxy values to array 'dataarray'
        
    print('Successfully compiled data!')
    print(str(len(dataarray))+' galaxies in dataset.')
    return dataarray

#plot positions of all galaxies
def part2(dataarray):
    plt.figure(figsize=(8,8))
    plt.title('Figure 1: All Galaxies',fontsize=14)
    plt.ylabel('Declination $(degrees)$',fontsize=12)
    plt.xlabel('Right Ascension $(degrees)$',fontsize=12)
    plt.xlim(193.5,196.5)
    plt.ylim(26.5,29.5)
    plt.gca().invert_xaxis()
    
    #plot individual coordinates
    for i in range(0,len(dataarray)):
        plt.plot(dataarray[i][1],dataarray[i][2],color='black',marker='.')
    plt.grid()
    return

#calculate and plot weighted mean of galaxy positions
def part3(dataarray):
    
    #compute mean Right Ascension
    sumnum = 0
    sumden = 0
    for i in range(0,len(dataarray)):
        w = 10**(dataarray[i][4]/2.5) #set weighting factor with r-band mag uncertainty
        sumnum += dataarray[i][1]/w #summation of numerator 
        sumden += 1/w #summation of denominator
    meanra = sumnum/sumden

    #compute mean Declination
    sumnum = 0
    sumden = 0
    for i in range(0,len(dataarray)):
        w = 10**(dataarray[i][4]/2.5)
        sumnum += dataarray[i][2]/w
        sumden += 1/w
    meandec = sumnum/sumden
    
    #Add mean RA,DEC to plot
    plt.plot(meanra,meandec,color='red',marker='*',markersize=14,label='Mean',linestyle='None')
    plt.legend(loc='upper center',ncol=1,fontsize=10,numpoints=1)
    pp.savefig()
    plt.clf()
    
    print('Found mean RA,Dec: ',meanra,meandec)
    return meanra,meandec

#Create historgram of redshift dist. and fit gaussian 
def part4(dataarray):

    #create list of all redshifts
    redshift = []
    for i in range(0,len(dataarray)):
        redshift.append(dataarray[i][6])

    #Gather data in bins of width 0.001. Retrive counts from each bin. 
    histogram = py.hist(redshift,bins=np.arange(0.005, 0.05, 0.001),color='lightgrey')
    counts = histogram[0]
    shifts = histogram[1]
    binset = []

    #Create list of z values at bin centers
    for i in range(0,len(shifts)-1):
        binset.append(shifts[i]+0.0005)

    #Define Gaussian function. 
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    #Get initial parameters for fitting (mean,stddev). 
    stddev = np.std(redshift)
    mean = np.mean(redshift)

    #Perform Gaussian fitting (coeffs = [peak,mean(z),stddev(sigma)]
    coeffs, pcov = optimize.curve_fit(gauss, binset, counts, p0 = [1, mean, stddev])       
    
    #Plot histogram and gaussian.
    plt.title('Figure 2: Redshifts of All Galaxies')
    plt.xlabel('Redshift $(z)$', fontsize = 12)
    plt.ylabel('Number of galaxies', fontsize = 12)
    plt.xlim(0,0.05)
    plt.plot(binset, gauss(binset, *coeffs),color='black',linestyle='-')
    plt.annotate(r'$\bar z$'+' ='+str('{:.4E}'.format(coeffs[1]))+'\n'+r'$\sigma_z$'+' ='+str('{:.4E}'.format(abs(coeffs[2]))), xy=(0.70, 0.90), xycoords='axes fraction',fontsize = 10)
    pp.savefig()
    plt.clf()

    print('Gaussian fit: peak,mean,sigma',coeffs)
    return coeffs #return gaussian fitting

#Compute recessional velocity and distance from Hubble's law.
def part5(gausscoeffs,dataarray):

    #Compute recessional velocity 
    zcoma = gausscoeffs[1]
    sigmacoma = gausscoeffs[2]
    c = 2.99792485e8
    vcoma = c*(((zcoma+1)**2)-1)/(((zcoma+1)**2)+1)
    vcoma = vcoma/1000

    #Compute distance from Hubble law
    H = 67.8
    dcoma = vcoma/H
    
    print('Coma radial velocity: '+str(vcoma)+'km/s')
    print('Coma distance (from Hubble law): '+str(dcoma)+'Mpc')
    return dcoma,zcoma #return cluster distance and mean redshift

#Define function for computing angular distance between two points
def projecteddistance(srcRA,srcDEC,comaRA,comaDEC,dcoma):

    #convert degrees to radians
    srcRArad = np.pi*srcRA/180
    srcDECrad = np.pi*srcDEC/180
    comaRArad = np.pi*comaRA/180
    comaDECrad = np.pi*comaDEC/180

    #Computation of angular distance  
    term1 = (np.sin(srcDECrad)*np.sin(comaDECrad))
    term2 = (np.cos(srcDECrad)*np.cos(comaDECrad)*np.cos((srcRArad)-(comaRArad)))
    combin = round(term1+term2,15) #round to 15 places
    radians = abs(np.arccos(combin))
    return radians

#Coma Cluster member selection algorithm
def part678(comaRA,comaDEC,dcoma,dataarray,gausscoeffs,zcoma):

    #compute mean RA and DEC of current galaxy set
    def part8a(dataarray):
        #compute mean Right Ascension
        sumnum = 0
        sumden = 0
        for i in range(0,len(dataarray)):
            w = 10**(dataarray[i][4]/2.5)
            sumnum += dataarray[i][1]/w
            sumden += 1/w
        meanra = sumnum/sumden

        #compute mean Declination
        sumnum = 0
        sumden = 0
        for i in range(0,len(dataarray)):
            w = 10**(dataarray[i][4]/2.5)
            sumnum += dataarray[i][2]/w
            sumden += 1/w
        meandec = sumnum/sumden
        
        print('Found mean RA,Dec: ',meanra,meandec)
        return meanra,meandec

    #perform gaussian fitting of current galaxy set
    def part8b(dataarray):
        
        #create list of all redshifts
        redshift = []
        for i in range(0,len(dataarray)):
            redshift.append(dataarray[i][6])

        #Gather data in bins of width 0.001. Retrive counts from each bin. 
        histogram = py.hist(redshift,bins=np.arange(0.005, 0.05, 0.001))
        counts = histogram[0]
        shifts = histogram[1]
        binset = []

        #Create list of z values at bin centers
        for i in range(0,len(shifts)-1):
            binset.append(shifts[i]+0.0005)

        #Get initial parameter guesses (mean,stddev). Get gaussian fit.
        def gauss(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
        stddev = np.std(redshift)
        mean = np.mean(redshift)
        coeffs, pcov = optimize.curve_fit(gauss, binset, counts, p0 = [1, mean, stddev])       
        plt.clf()

        print('Gaussian fit: peak,mean,sigma',coeffs)
        return coeffs,mean,stddev

    #Compute recessional velocity and distance of current galaxy set.
    def part8c(coeffs,dataarray):
        
        #Compute recessional velocity 
        zcoma = coeffs[1]
        sigmacoma = coeffs[2]
        c = 2.99792485e8
        vcoma = c*(((zcoma+1)**2)-1)/(((zcoma+1)**2)+1)
        vcoma = vcoma/1000

        #Compute distance from Hubble law
        H = 67.8
        dcoma = vcoma/H
        
        print('Coma radial velocity: '+str(vcoma)+'km/s')
        print('Coma distance (from Hubble law): '+str(dcoma)+'Mpc')
        return dcoma,zcoma


    dataarray2 = [] #create array that will contain galaxies removed from Coma Cluster member set.

    ### SELECTION ALGOITHM ###
    j=0 #interation number
    cont=True
    while cont==True: #repeat iterations until loop broken
        dell=False #Set 'galaxy deleted' boolean to False. Convergence is reached when dell=False at the end of the interation.
        numdell = 0 #Reset number of galaxies deleted on this iteration to zero
        j+=1 #increase iteration number

        print('\nIteration '+str(j)+' of member group selection...')
        comaRA,comaDEC = part8a(dataarray) #calculate average RA,DEC
        coeffs,mean,stddev = part8b(dataarray) #fit gaussian 
        dcoma,zcoma = part8c(coeffs,dataarray) #get distance and average z
        threesigma = abs(coeffs[2]*3) #set 3 sigma limit      

        #check all galaxies against current values
        i = 0 
        while i<len(dataarray):
            
            #get radial distance from mean ra,dec 
            radians = projecteddistance(dataarray[i][1],dataarray[i][2],comaRA,comaDEC,dcoma)
            distance = dcoma*radians

            #remove galaxy if z outside of 3sigma range
            if abs(zcoma-dataarray[i][6])>abs(threesigma):
                dataarray2.append(dataarray[i]) #add to array of removed galaxies
                del (dataarray[i]) #delete from dataarray
                dell=True #set 'galaxy deleted' boolean to true. Algorithm will perform another iteration
                numdell+=1 #+1 to number of galaxies deleted on this iteration
            
            #remove galaxy if radial distance greater than 1.5 Mpc
            if distance>1.5:
                dataarray2.append(dataarray[i]) #add to array of removed galaxies
                del (dataarray[i]) #delete from dataarray
                dell=True #set 'galaxy deleted' boolean to true. Algorithm will perform another iteration
                numdell+=1 #+1 to number of galaxies deleted on this iteration

            else:
                i+=1 
                
        print('Removed '+str(numdell)+' galaxies on this interation...')

        #end selection algorithm if no galaxies deleted on iteration
        if dell==False:
            break

    #compute mean z of member galaxies
    sumnum = 0
    sumden = 0
    for i in range(0,len(dataarray)):
        sumnum += dataarray[i][6]/dataarray[i][7]
        sumden += 1/dataarray[i][7]
    comameanz = sumnum/sumden
    
    print('\nMember group: '+str(len(dataarray))+' galaxies')
    print('Right Ascension: '+str(comaRA)+' Declination: '+str(comaDEC))
    print('Mean redshift: '+str(comameanz))
    print('Distance (Hubble\'s law): '+str(dcoma)+'Mpc')
    
    return dataarray,dataarray2,dcoma,comaRA,comaDEC,coeffs,comameanz

#Plot member galaxies, non-member galaxies, cluster center, redshift distribution, and gaussian fitting
def part9(dataarray,dataarray2,comaRA,comaDEC,dcoma,comameanz):

    #Plot galaxies and cluster center
    plt.title('Figure 3: Coma Cluster Galaxies',fontsize=14)
    plt.ylabel('Declination '+'$ra\,(degrees)$',fontsize=12)
    plt.xlabel('Right Ascension '+r'$dec\,(degrees)$',fontsize=12)
    plt.xlim(193.5,196.5)
    plt.ylim(26.5,29.5)
    for i in range(0,len(dataarray2)):
        plt.plot(dataarray2[i][1],dataarray2[i][2],markeredgecolor='black',marker='o',markerfacecolor='None',markersize=3)
    for i in range(0,len(dataarray)):
        plt.plot(dataarray[i][1],dataarray[i][2],color='black',marker='o',markersize=3,fillstyle=None)
    plt.plot(comaRA,comaDEC,color='red',marker='*',markersize=14,label='Center',linestyle='None')
    plt.plot(0,0,color='black',marker='o',markersize=3,fillstyle=None,label='Member',linestyle='None')
    plt.plot(0,0,markeredgecolor='black',marker='o',markerfacecolor='None',markersize=3,label='Non-member',linestyle='None')
    
    #Plot ellipse
    ratio = 1/np.cos(np.pi*comaDEC/180) #compute major axis distortion at declination of cluster center
    ellipseA = ((1.5*180)/(np.pi*dcoma))*ratio #set major axis
    ellipse = Ellipse((comaRA,comaDEC),width=2*(ellipseA),height=2*(1.5*180)/(dcoma*np.pi),linestyle=':',fill=False)
    fig = plt.gcf()
    ax = fig.gca()
    plt.legend(loc='upper center',ncol=3,fontsize=10,numpoints=1)
    plt.gca().invert_xaxis()
    ax.add_patch(ellipse)
    plt.grid()
    pp.savefig()
    plt.clf()

    #create list of all redshifts
    redshift = []
    redshift2 = []
    for i in range(0,len(dataarray)):
        redshift.append(dataarray[i][6])
        redshift2.append(dataarray[i][6])
    for i in range(0,len(dataarray2)):
        redshift2.append(dataarray2[i][6])

    #Gather data in bins of width 0.001. Retrive counts from each bin.
    histogram = py.hist(redshift,bins=np.arange(0.005, 0.05, 0.001),color='lightgrey')
    counts = histogram[0] #get counts
    shifts = histogram[1] #get z values of bins
    binset = []

    #Create list of z values at bin centers
    for i in range(0,len(shifts)-1):
        binset.append(shifts[i]+0.0005)

    #Define Gaussian, calculate initial parameter guesses.
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    stddev = np.std(redshift)
    mean = np.mean(redshift)

    #Fit gaussian
    coeffs, pcov = optimize.curve_fit(gauss, binset, counts, p0 = [1, mean, stddev])       

    #Plot histogram and fitted Gaussian.
    plt.title('Figure 4: Coma Cluster Member Galaxy Redshifts')
    plt.xlabel('Redshift '+r'$z$', fontsize = 12)
    plt.ylabel('Number of galaxies', fontsize = 12)
    plt.xlim(0,0.05)
    plt.plot(binset, gauss(binset, *coeffs),color='black',linestyle='-')
    plt.annotate(r'$\bar z$'+' ='+str('{:.4E}'.format(comameanz))+'\n'+r'$\sigma_z$'+' ='+str('{:.4E}'.format(abs(coeffs[2]))), xy=(0.70, 0.90), xycoords='axes fraction',fontsize = 10)
    pp.savefig()
    plt.clf()
    
    return


def part1011(dataarray,dcoma,comaRA,comaDEC):
    
    #convert coma distance to parsecs
    dcomaparsecs = dcoma*10**6
    
    for i in range(0,len(dataarray)):

        #Newly compute values appended to rows of dataarray as follows...
        #[objID(0),ra(1),dec(2),mag(3),magErr(4),ext(5),z(6),zErr(7),magabs(8),magabserr(9),lum(10),distance(11)]

        #calculate absolute magnitude and error
        mapp = (dataarray[i][3]-dataarray[i][5])
        magabs = mapp - 5*np.log10(dcomaparsecs/10)
        magabserr = dataarray[i][4]
        dataarray[i].append(magabs)
        dataarray[i].append(magabserr)
        
        #calculate solar lumonisties
        lumin = 10**((4.76-magabs)/2.5)
        dataarray[i].append(lumin)
        
        #calculate radial distance from centre of group
        radians = projecteddistance(dataarray[i][1],dataarray[i][2],comaRA,comaDEC,dcoma)
        distance = dcoma*radians
        dataarray[i].append(distance)
        
    return dataarray

#Plot CDF 
def part1213(dataarray):

    #Plot galaxies as radius vs. luminosity
    plt.title('Figure 5: Coma Cluster Member Galaxy Luminosities')
    plt.xlim(0,1.5)
    plt.yscale('log')
    plt.xlabel(r'Radial Distance '+r'$R\,(Mpc)$')
    plt.ylabel(r'Luminosity '+r'$L_{gal}\,(10^{10}L_\odot)$')
    for l in range(0,len(dataarray)):
        plt.plot(dataarray[l][11],dataarray[l][10]/(10**10),color='black',marker='o',markersize=3,fillstyle=None)
    pp.savefig()
    plt.clf()
    
    #Calculate total luminosity
    totallum=0
    for i in range(0,len(dataarray)):
        totallum += dataarray[i][10]

    #caculate cummulative luminosity distribution in increments
    sumlumin = [] #store sum of luminosities at increments of R   
    radials = np.arange(0,1.51,0.02) #Increment R 75 times from 0-1.5Mpc
    #Compute CDF 
    for i in range(0,len(radials)):
        lumin=0
        for j in range(0,len(dataarray)):
            if dataarray[j][11]<radials[i]:
                lumin += dataarray[j][10]
        sumlumin.append(lumin/totallum)

    #Format plot
    plt.title('Figure 6: Coma Cluster Cumulative Luminosity Distribution')
    plt.xlim(0,1.5)
    plt.ylim(0,1)
    plt.xlabel(r'Radial Distance '+r'$R\,(Mpc)$')
    plt.ylabel(r'Normalized Luminosity '+r'$L_{total}\,(3.94\ast10^{10}L_\odot)$')

    #Define expontial function to fit to CDF
    def func(x,a,b,c):
        return a*x**b + c
    #Fit CDF to func
    cdf,pcov = optimize.curve_fit(func, radials, sumlumin, p0 = [0, 1, 0])
    print('CDF fit [y=ax^b+c]: '+str(cdf))

    #calculate Reff,r1/2
    reff = ((0.5-cdf[2])/cdf[0])**(1/cdf[1])
    rhalf = 4*reff/3
    
    print('Reff: '+str(reff)+' Mpc')
    print('r1/2: '+str(rhalf)+' Mpc')
    print('L total: '+str(totallum)+' solar luminosities')

    #plot CDF, CDF fit. Compute and add Reff, r1/2, and L1/2 as short lines
    x = np.arange(0,1.5,0.01)
    plt.plot(x,func(x,*cdf),linestyle='--',color='black')
    plt.plot(radials,sumlumin,color='black',marker=None,markersize=3,fillstyle=None)
    plt.axvline(x=reff, ymin=0.4, ymax = .6, linewidth=1, color='black',linestyle=':')
    plt.axvline(x=rhalf, ymin=func(rhalf,*cdf)-.09, ymax = func(rhalf,*cdf)+.1, linewidth=1, color='black',linestyle=':')
    plt.axhline(y = func(rhalf,*cdf), xmin=rhalf-0.4, xmax=rhalf-0.21, linewidth=1, color='black',linestyle=':')
    plt.annotate('$R_{eff}$',(reff+0.01,0.42))
    plt.annotate('$r_{1/2}$',(rhalf+0.01,func(rhalf,*cdf)-.08))
    plt.annotate('$L_{1/2}$',(rhalf+0.16,func(rhalf,*cdf)))
    pp.savefig()
    plt.clf()
    
    return cdf,reff,rhalf,totallum

#Calculate recessional velocity for a redshift z
def velocity(z):
    c = 2.99792458E5
    numer = (((z+1)**2)-1)*c
    denom = (((z+1)**2)+1)
    v = numer/denom
    return v

#Format of dataarray rows at this point:
#[objID(0),ra(1),dec(2),mag(3),magErr(4),ext(5),z(6),zErr(7),magabs(8),magabserr(9),lum(10),distance(11),peculairv(12)]
def part1415(dataarray,dcoma,comameanz):

    #get recessional velocity
    comav = velocity(comameanz)

    #Compute recessional velocities of individual galaxies
    for i in range(0,len(dataarray)):
        recev = velocity(dataarray[i][6])
        pecv = comav-recev
        dataarray[i].append(pecv)

    #create list of all peculiar velocities
    pecvslist = []
    for i in range(0,len(dataarray)):
        pecvslist.append(dataarray[i][12])
    
    #Gather data in bins of width 200km/s. Retrive counts from each bin. 
    histogram = py.hist(pecvslist,bins=np.arange(-3000, 3000, 200))
    counts = histogram[0]
    binvalues = histogram[1]
    binset = []

    #Create list of peculiar velocities at bin centers
    for i in range(0,len(binvalues)-1):
        binset.append(binvalues[i]+100)

    #Get initial parameter guesses (mean,stddev). Get gaussian fit.
    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    stddev = np.std(pecvslist)
    mean = np.mean(pecvslist)
    coeffs, pcov = optimize.curve_fit(gauss, binset, counts, p0 = [50, mean, stddev])
    
    print('Gaussian fit: '+str(coeffs))
    print('Velocity dispersion: '+str(coeffs[2])+'km/s')
    
    #Plot histogram and gaussian curve.
    plt.title('Figure 7: Coma Cluster Member Galaxy Peculiar Velocities')
    plt.xlabel('Peculiar Velocity '+r'$v_{pec}\,(km/s)$', fontsize = 12)
    plt.ylabel('Number of galaxies', fontsize = 12)
    plt.xlim(-3000,3000)
    plt.ylim(0,50)
    plt.hist(pecvslist,bins=np.arange(-3000, 3000, 200),color='lightgrey',label='Member')
    plt.plot(binset, gauss(binset, *coeffs),color='black',linestyle='-')
    pp.savefig()
    plt.clf()
    return coeffs,dataarray

#Compute M1/2,L1/2, M/L
def part161718(cdf,coeffs,reff,rhalf,totallum):
    mhalf = 3*((coeffs[2]/1.023)**2)*(rhalf*10**6)/(4.5*10**-3)
    print('Virial mass M1/2: '+str(mhalf)+' M(solar)')
    lhalf = totallum*(cdf[0]*rhalf**cdf[1] + cdf[2])
    print('L1/2: '+str(lhalf)+' L(solar)')
    mlratio = mhalf/lhalf
    print('M/L: '+str(mlratio))
    return

######MAIN PROGRAM EXECUTION######
print('-----------------------------------------------------------------')
print('                    COMA CLUSTER ANALYSIS')
print('-----------------------------------------------------------------') 
dataarray = readData(filename)
print('')
part2(dataarray)
print('')
comaRA,comaDEC = part3(dataarray)
print('')
gausscoeffs = part4(dataarray)
print('')
dcoma,zcoma = part5(gausscoeffs,dataarray)
print('')
dataarray,dataarray2,dcoma,comaRA,comaDEC,coeffs,comameanz = part678(comaRA,comaDEC,dcoma,dataarray,gausscoeffs,zcoma)
print('')
part9(dataarray,dataarray2,comaRA,comaDEC,dcoma,comameanz)
print('')
dataarray = part1011(dataarray,dcoma,comaRA,comaDEC)
print('')
cdf,reff,rhalf,totallum = part1213(dataarray)
print('')
coeffs,dataarray = part1415(dataarray,dcoma,comameanz)
print('')
part161718(cdf,coeffs,reff,rhalf,totallum)

pp.close()
sys.exit()
