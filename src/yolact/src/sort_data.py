import numpy as np


class Sort_data:
    #----------------------list--------------------------------------
    # 0: ID        3: mask picture(now)    6: predict point     
    # 1: name      4: box point
    # 2: scorce    5: three mask pictures 
    #----------------------------------------------------------------
    def __init__(self):
        self.data_size = 20
        self.old_dataset = [[ [] for i in range(0)] for j in range(self.data_size)]
        self.origin_data = [[ [] for i in range(0)] for j in range(self.data_size)]
        self.old_obj_count = 0
        self.total_count = 0
        for i in range(self.data_size):
            #---------------------ID---------------------
            self.old_dataset[i] = [i] 
            self.origin_data[i] = [i]
            #--------------------------------------------
            for k in range(3):
                self.old_dataset[i].append(0)
                self.origin_data[i].append(0)

            self.old_dataset[i].append([0,0,0,0,0,0])
            self.origin_data[i].append([0,0,0,0,0,0])

            for j in range(2):   
                self.old_dataset[i].append([]) 
                self.origin_data[i].append([])
      
    def Compare_Data(self, old_data, new_data, old_count, new_count):
        num = 0
        list_info = []
        data_set = [[ [] for i in range(0)] for j in range(self.data_size)]
        for i in range(self.data_size):
            data_set[i] = [i] 
            for k in range(3):
                data_set[i].append(0)
            data_set[i].append([0,0,0,0,0,0])
            for j in range(2):   
                data_set[i].append([]) 


        for i in range(new_count):
            distance = []
            for j in range(old_count):
                distance.append(np.sqrt( pow((new_data[i][4][0] - old_data[j][4][0]),2) +  pow((new_data[i][4][1] - old_data[j][4][1]),2) ))

            if distance != []:
                min_id = distance.index(min(distance))

                if distance[min_id] < 20:
                    if new_data[i][1] == old_data[min_id][1]:   
                        #currect object
                        
                        data_set[min_id] = new_data[i]
                        data_set[min_id][0] = min_id
                        data_set[min_id][5] = old_data[min_id][5]
                        data_set[min_id][6] = old_data[min_id][6]
                        
                    else:
                        #new object
                        data_set[self.data_size-1-num] = new_data[i]
                        data_set[self.data_size-1-num][0] = self.data_size-1-num
                        num+=1

                else: 
                    #other the same object, origin object is disapear
                    data_set[self.data_size-1-num] = new_data[i]
                    data_set[self.data_size-1-num][0] = self.data_size-1-num
                    num+=1 

        if num != 0: 
            for i in range(self.data_size):
                if data_set[i][1] == 0:
                    list_info.append(i)
            for i in range(num):
                #change obj info
                data_set[list_info[i]], data_set[self.data_size-1-i] = data_set[self.data_size-1-i], data_set[list_info[i]] 
                data_set[list_info[i]][0], data_set[self.data_size-1-i][0] = data_set[self.data_size-1-i][0], data_set[list_info[i]][0] 
        
        if new_count < old_count:
            total_obj = old_count  
        total_obj = new_count 

        new_data = data_set
        return new_data, total_obj


    def data_save(self, mask_picture, classes, names, scorces, boxes, first_flag, old_obj_info):
        self.compare_data = [[ [] for i in range(0)] for j in range(self.data_size)]
        for i in range(self.data_size):
            self.compare_data[i] = [i]
            for k in range(3):
                self.compare_data[i].append(0)
            self.compare_data[i].append([0,0,0,0,0,0])  
            for j in range(2):   
                self.compare_data[i].append([])

        new_obj_count = 0
        if first_flag == False:
            for i in range(len(classes)):
                self.origin_data[i][1] = names[i]
                self.origin_data[i][2] = scorces[i]
                self.origin_data[i][3] = mask_picture[i]
                x1,y1,x2,y2 = boxes[i, :]
                self.origin_data[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
                self.old_obj_count+=1
            new_data = self.origin_data
        else:
            for i in range(len(classes)):
                self.compare_data[i][1] = names[i]
                self.compare_data[i][2] = scorces[i]
                self.compare_data[i][3] = mask_picture[i]
                x1,y1,x2,y2 = boxes[i, :]
                self.compare_data[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
                new_obj_count+=1
            new_data, self.total_count = self.Compare_Data(old_obj_info, self.compare_data, self.old_obj_count, new_obj_count)    
            #self.origin_data = new_data
            self.old_obj_count = new_obj_count

        return new_data, self.total_count

    #def data_MaskImaage_save(self, data_info, mask_image):
        
        

if __name__ == '__main__':
    Sort_data()