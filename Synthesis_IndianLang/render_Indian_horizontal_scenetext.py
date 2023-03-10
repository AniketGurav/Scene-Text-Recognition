# -*- coding: UTF-8 -*-
#requires imagemagick, pango, cairo, pangocairo installed on your linux machine
# this python script invokes bash commands 


# the GT of the image is also written along with the command generation
#two gt files will be generated ; one with simple image path <space> gt and another an xml with all details of rendering like
#  The code is ineffecient for the reason that convert commands would be called multiple times and multiple read/write operations takes place . Bettering this involves use of clone/mpr options of convert !!
#########

#################### NO BLEND MODE #####################################

#### it was observed that blending fg and bg layers with natural images generate much complicated images than the usual ones we see
## most scene images just have plain colors in fg and bg ; hence this code just makes such images ###

# read alpha_readme.txt for the issues / alternatives in this regard

####
#### arguments list #
#argv[1] -- vocab file with full path
#argv[2] -- list of unique fonts to be used

# argv[3] -- parent dir where the rendered images will be written to
# argv[4] -- languageName ; eg Arabic, Malayalam
# argv[5] -- iterationNumber , when the rendering is ran multiple times
# argv[6] -- dir path for the pictures for the background


import codecs,subprocess,os,sys,glob
import random
import PIL
from PIL import Image
from subprocess import call
random.seed()
#language='Arabic'
#process='intialRender'
#iteration="One"
#get the list of unique font family Names
with open(sys.argv[2]) as f:
	fontsList = f.read().splitlines()
print 'number of unique fonts being considered= ', len(fontsList)
#a set of images , whose random crops can be used as background for the rendered word images. We used Validation set of Places dataset for this
# replace the below path with your location of images
PlacesImList=glob.glob(sys.argv[6]+"/*.jpg")

writeDirParent=sys.argv[3]
xmlFileName=writeDirParent+'/label.txt'


#a flist of words separated by newline
vocabFile = codecs.open(sys.argv[1],'r',encoding='utf8')
myfile = codecs.open(xmlFileName,'a',encoding='utf8')
words = vocabFile.read().split()
distortArcOptions={'40','60','70','80','40','40','30'}
#skewOptions={'1','2','3','4','5','-1','-2','-3','-4','-5'}
distorArcBooleanOptions={0,0,0,0,0,0,0,0,1}
#skewBooleanOptions={0,0,0,0,0,1}
densityOptions={'100','150','150','150','200','200','200','250','250','300','300','200','150','200','300','300','300','250','250'}
boldBooleanOptions={0,0,1}
italicBooleanOptions={0,0,0,0,0,0,0,0,0,0,0,0,1}
fontSizeOptions={'12','14','16','18','20','22','24','26','28','32','34','36'}
fontSizeOptions={'44','46','56','58','66','72'}
trimOptions={0,1}
fontStretchOptions={ 'semicondensed', 'normal', 'semiexpanded',  'normal', 'normal', 'normal','normal','normal','normal', 'normal','normal','normal','semicondensed','semicondensed','semicondensed','semicondensed','semicondensed', 'semiexpanded', 'semiexpanded', 'semiexpanded', 'semiexpanded','normal', 'normal', 'normal','normal','normal','normal', 'normal','normal','normal'}
shadowBooleanOptions={0,0,0,0,0,0,0,0,1}
perspectiveBooleanOptions={0,0,0,0,0,1,1,1}

shadowWidthOptions={'0','0','0','2','3','4'}
shadowSigmaOptions={'1','3'}
shadowOpacityOptions={'100','100','100','100','90','80','70'}
shadowWidthSignOptions={'+','-'}
#outputfile = open('render_commands_'+language+'_'+process+'_'+iteration+'.sh','w')
#gtfile = open('ocr_gt.txt','w')

numWords=len(words)
print 'number of words in the vocab= ', numWords

#writeDir=writeDirParent+'0\/'
# number_of_alphabets = 4
# each_alphabet_repeated = 100
# total_iterations = (numWords - number_of_alphabets) + (number_of_alphabets * each_alphabet_repeated)
# alphabet_iter = 1
word_iter = 0

for i in range(1110000000,numWords+1110000000):
    if i%1000==0:
        print 'completed ', i
        # thousand=i/1000
    	writeDir=writeDirParent+'/'
    	#print writeDir
        if not os.path.exists(writeDir):
    	   os.makedirs(writeDir)
        filelist = glob.glob("*.png")#remove all temp png files after every 1000 words
        for f in filelist:
            os.remove(f)
    textImageName=str(i)+'_text.png'

    # to convert and rgb in tuple to rgb hex representation '#%02x%02x%02x' % (fg[0], fg[1], fg[2])
    fg=random.sample(range(50, 255), 3) ###### making fg color more brighter ##

    bg=random.sample(range(0, 255), 3)
    bg[0]=abs(fg[0]+100-255)
    bg[1]=abs(fg[1]+100-255)
    bg[2]=abs(fg[2]+125-255)
    sd=random.sample(range(0, 255), 3)

    fg_hex='#%02x%02x%02x' % (fg[0], fg[1], fg[2])
    bg_hex= '#%02x%02x%02x' % (bg[0], bg[1], bg[2])
    sd_hex= '#%02x%02x%02x' % (sd[0], sd[1], sd[2])

    if bool(random.getrandbits(1)):
        tmp=fg_hex
        fg_hex=bg_hex
        bg_hex=tmp
	
    ## random density skew slant font fontsize kerning

    density=random.sample(densityOptions,1)[0]
    distortArcBoolean=random.sample(distorArcBooleanOptions,1)[0]
    boldBoolean=random.sample(boldBooleanOptions,1)[0]
    italicBoolean=random.sample(italicBooleanOptions,1)[0]
    fontSize=random.sample(fontSizeOptions,1)[0]
    fontName=random.sample(fontsList,1)[0]
    fontStretch=random.sample(fontStretchOptions,1)[0]

    shadowOpacity=random.sample(shadowOpacityOptions,1)[0]
    shadowSigma=random.sample(shadowSigmaOptions,1)[0]
    ShadowWidth=random.sample(shadowWidthOptions,1)[0]
    ShadowWidthSign=random.sample(shadowWidthSignOptions,1)[0]
    ### making the convert command ####

    command='convert  -alpha set  -background none'
    skewValue='0'
    arcValue='0'
    if distortArcBoolean==1:
    	distortArc=random.sample(distortArcOptions,1)[0]
    	command+=' -distort Arc '+ distortArc
    	arcValue=distortArc
    command+=' pango:\'   <span '
    command+='font_stretch='+'\"'+fontStretch+'\" '
    command+='foreground='+'\"'+fg_hex+'\" '

    ##selecting the word
    # if len(words[word_iter]) > 1:
    #     textWord = words[word_iter]
    #     word_iter += 1
    # else:
    #     if alphabet_iter % each_alphabet_repeated == 0:
    #         textWord = words[word_iter]
    #         alphabet_iter = 1
    #         word_iter += 1
    #     else:
    #         textWord = words[word_iter]
    #         alphabet_iter += 1
    
    # word_to_print = textWord
    # if word_iter == numWords:
    #     word_iter -= 1
    textWord = words[word_iter]
    word_iter += 1
    word_to_print = textWord
    # l = len(textWord)
    if word_iter == numWords:
        word_iter -= 1
        
    if italicBoolean==1:
    	textWord='<i>'+textWord+'</i>'
    if boldBoolean==1:
    	textWord='<b>'+textWord+'</b>'
    #fontName='Roboto Black'
    fontString='font='+'\"'+fontName+' '+fontSize+' \">  '
    fontString+=' '+ textWord + '</span> \''
    command+=fontString
    #command+=' rendered_image.jpeg'
    trimBoolean=random.sample(trimOptions,1)[0]
    #command+=' -trim ' #do trim in all cases
    command+=' png:-|'

    #### add shadow/border ######

    command+='convert - ' + ' \\( +clone -background ' + '\''+str(sd_hex)+'\' -shadow '
    command+= shadowOpacity+'x'+shadowSigma+ShadowWidthSign+ShadowWidth+ShadowWidthSign+ShadowWidth + ' \\) +swap  -background none   -layers merge  +repage '+ 'png:-| '

    ######  distort the perspective of the image ########
    perspectiveBoolean=random.sample(perspectiveBooleanOptions,1)[0]
    # if len(words[word_iter]) > 1:
    if perspectiveBoolean==1:
        sx=random.uniform(0.7, 1.3)
        ry=random.uniform(-0.8, 0.8)
        rx=random.uniform(-0.15, 0.15)
        sy=random.uniform(0.7, 1.3)
        px=random.uniform(0.0001, 0.001)
        py=random.uniform(0.0001, 0.001)
        command+='convert - ' + ' -alpha set -virtual-pixel transparent +distort Perspective-Projection '
        command+= '\''+str(sx)+ ', ' + str(ry) + ', 1.0\t' + str(rx) + ', ' + str(sy) + ', 1.0\t' + str(px) + ', ' + str(py) + '\'  png:-| '
    # else:
        # px = py = 0
        # control = float((alphabet_iter % each_alphabet_repeated) / float(each_alphabet_repeated/2))
        # rx = -0.15 + (control * 0.15)
        # ry = -0.8 + (control * 0.8)
        # sx = 0.7 + (control * 0.3)
        # sy = 0.7 + (control * 0.3)
        # px = 0.0001 + (control * 0.00045)
        # py = 0.0001 + (control * 0.00045)

        # rotate_image = -90 + float(control * 90)
        # print(control,rotate_image)
        
        # command+='convert - ' + '-alpha set -virtual-pixel transparent -rotate ' + str(rotate_image) + ' png:-| '

            # sin='convert xc: -format "%[fx:sin( '+str(rotate_image)+' *pi/180)]" info:'
            # cos='convert xc: -format "%[fx:cos( '+str(rotate_image)+' *pi/180)]" info:'
            # cx=37; cy=37;   # center of rotation
            # command+= 'convert -virtual-pixel Black  -interpolate NearestNeighbor -fx "ii = i - '+str(cx)+';   jj = j - '+str(cy)+';'
            # command+= 'xx =  '+str(cos)+'*ii +'+str(sin)+'*jj + '+str(cx)+'; yy = -'+str(sin)+'*ii +'+str(cos)+'*jj + '+str(cy)+'; v.p{xx,yy}" '
        # command+='convert - ' + ' -alpha set -virtual-pixel transparent +distort Perspective-Projection '
        # command+= '\''+str(sx)+ ', ' + str(ry) + ', 1.0\t' + str(rx) + ', ' + str(sy) + ', 1.0\t' + str(px) + ', ' + str(py) + '\'  png:-| '

    command+= ' convert - '
    if trimBoolean==1:
        command+='  -trim '
    #command+=' png:-|'
    command+=' -resize x32 '
    command+=textImageName

    #print '*******'
    #print command.encode('utf-8')
    os.system(command.encode('utf-8'))
    finalFgLayerName=textImageName
    im=Image.open(textImageName)
    imWidth, imHeight = im.size

    #print bgCommand.encode('utf-8')
    #convert -background none -alpha set -channel A -evaluate set 100%   -size "88x50" xc:yellow  yellow100.png
    ####### blend fg and bg with a natural scene images separately ##########
    #fgBlendBooleanOptions={0,0} #should a natural image be blended with the text stroke ie the fg layer
    #bgBlendBooleanOptions={0,0} #should the bg (which is a uniform color for now) be blended with a natural scene
    #fgBlendBoolean=random.sample(fgBlendBooleanOptions,1)[0]
    #bgBlendBoolean=random.sample(bgBlendBooleanOptions,1)[0]
    #################################################################################################################
    # to make the rendering simpler, the blending part is only done in this manner				#
    # 1. blend the fg with a natural image and keep the bg a uniform color itself				#
    # 2. keep fg color uniform and overlay it on a natural image ( ie the bg is a crop from a natural image)	#
    #################################################################################################################
    fgorBgBooleanOptions={0,1} # 0 means text should be blended 1 means bg is just a natural image crop
    fgOrBgBoolean=random.sample(fgorBgBooleanOptions,1)[0]
    # pick a random image from places dataset and get a crop from it, of the same size as our textImage
    naturalImageName=random.sample(PlacesImList,1)[0]
    fgImage=Image.open(naturalImageName)
    fgWidth, fgHeight = fgImage.size
    if fgWidth < imWidth+5 or  fgHeight < imHeight+5:
        fgImage=fgImage.resize((imWidth+10,imHeight+10),PIL.Image.ANTIALIAS)
        fgWidth, fgHeight = fgImage.size

                #get a random crop from the image chosen from blending
    x=random.sample(range(0,fgWidth-imWidth+2 ),1)[0]
    y=random.sample(range(0, fgHeight-imHeight+2),1)[0]
    w=imWidth
    h=imHeight
    fgImageCrop=fgImage.crop((x ,y ,x+w, y+h))
    fgImageCropName=str(i) + '_naturalImage.png'
    fgImageCrop.save(fgImageCropName)
    fgBlendBoolean=0
    bgNaturalImage=0
    #print fgOrBgBoolean	
    if fgOrBgBoolean==0:
        #fgComposeModeOptions={'Multiply','over','Dst_out', 'Screen', 'Bumpmap', 'Divide', 'plus', 'minus', 'ModulusAdd', 'difference'  }
        #fgComposeMode=random.sample(fgComposeModeOptions,1)[0]

        textBgBlendImageName=str(i)+'_textBgBlend.png' #the image which is blend of natural image and the fg text layer
        textBgBlendCommand='composite ' + textImageName + ' -compose Dst_in ' + fgImageCropName + ' -alpha set ' +  textBgBlendImageName

        #textBgBlendCommand='composite ' + textImageName +  ' -compose ' + fgComposeMode + ' ' + fgImageCropName + ' -alpha set ' + textBgBlendImageName
        
        os.system(textBgBlendCommand.encode('utf-8'))
        finalFgLayerName=textBgBlendImageName
        ####  now in this case our bg wll be a uniform color ####
        bgCommand='convert -background none -alpha set -size '
        bgCommand+='\"' + str(imWidth) + 'x' + str(imHeight) + '\" xc:' + '\"' + bg_hex + '\" '
        bgLayerName=str(i)+'_bgLayer.png'
        bgCommand+= bgLayerName

        #print bgCommand.encode('utf-8')
        os.system(bgCommand.encode('utf-8'))
        fgBlendBoolean=1
 
    else:
        #when fgorBgBoolean==1 we dont to fg blend but choose bg as a ntural image crop
	#print 'bgisnatural'
        bgLayerName=fgImageCropName
        bgNaturalImage=1
    #finalFgLayerName=textImageName
    finalBgLayerName=bgLayerName




    #### combine fglayer and bglyaer in normal manner  - just do normal overlay or say dissovlve method ###
    finalBlendImageName=writeDir+str(i)+'.jpeg'
    finalBlendCommand='composite -gravity center ' + finalFgLayerName + ' ' +  finalBgLayerName
    finalBlendCommand+=' jpeg:-|'
    finalBlendCommand+='convert -  ' + finalBlendImageName



    #print finalBlendCommand.encode('utf-8')
    os.system(finalBlendCommand.encode('utf-8'))



    ### WRITING THE GT ALONG WITH RENDERING DETAILS TO a csv ###
    #print command.encode('utf-8')
    #print command_compose
    myfile.write(str(i)+'.jpeg ')
    myfile.write(word_to_print + '\n')
    # myfile.write(str(i) + ', ')
    # myfile.write(fontName + ', ')
    # myfile.write(fontSize + ', ')
    # myfile.write(fontStretch + ', ')
    # myfile.write(arcValue + ', ')
    # myfile.write(str(perspectiveBoolean) + ', ')
    # myfile.write(str(fgBlendBoolean) + ', ')
    # myfile.write(str(bgNaturalImage) + ',\n')

myfile.close()
