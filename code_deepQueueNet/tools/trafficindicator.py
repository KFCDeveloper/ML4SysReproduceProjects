# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache-2.0 License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache-2.0 License for more details.

import numpy as np 
 


    
    
    
class feaTure:
    def __init__(self, inflow, no_of_port, no_of_buffer, WINDOW, ser_rate):
        self.inflow=inflow
        self.no_of_port=no_of_port 
        self.no_of_buffer=no_of_buffer
        # 表示每个时间段内的“服务时间”，即在这段时间内平均处理一个数据包所需的时间。这个时间是根据数据包长度以及服务速率来计算的。
        self.tau_list=(inflow['pkt len (byte)'].rolling(WINDOW, min_periods=1).mean()/ser_rate).values    
    
    def getNum(self, Time, i, TAU):
        t=Time[i] #current time 
        no=0
        ix=i-1
        while ix>=0:
            if t-Time[ix]<=TAU:
                no+=1
                ix-=1
            else:
                break 
        return no 
 
 
    # C_dst_SET 其对应的值是一个列表，列表中存储了在某个时间段内到达端口 i 且优先级为 j 的流的数量。(这个时间段是Timer中记录的两个时刻之间的时间段)
    # 这个时间段是根据输入的时间戳数据和 tau_list 计算得出的。这个字典用于记录不同端口和缓冲区的流量信息。
    # TAU 是 对应时间的work loads
    # LOAD : LOAD[i] 表示端口编号为 i 的端口的平均负载。负载是指在每个时间段内到达该端口的流的数量的平均值。
    def getCount(self):
        Time=self.inflow['timestamp (sec)'].values
        SRC=self.inflow['src'].values
        DST=self.inflow['dst'].values
        Prio=self.inflow['priority'].values
        C_dst_SET={(i,j):[0] for i in range(self.no_of_port) for j in range(self.no_of_buffer)}
        LOAD={i: [0.] for i in range(self.no_of_port)}

        for t in range(1, len(Time)):
            TAU=self.tau_list[t]  #adapted service time to cal. the corresponding work loads. 
            ix=self.getNum(Time, t, TAU)
            src=SRC[t-ix:t] if ix>0 else np.array([])
            dst=DST[t-ix:t] if ix>0 else np.array([])
            prio=Prio[t-ix:t] if ix>0 else np.array([])

            for i in range(self.no_of_port):
                LOAD[i].append(np.sum(src==i))
                for j in range(self.no_of_buffer):
                    C_dst_SET[(i,j)].append(np.sum((dst==i) & (prio==j)))  

        for i in range(self.no_of_port):
            LOAD[i]=np.mean(np.array(LOAD[i])[SRC==i])  if len(np.array(LOAD[i])[SRC==i])>0 else 0.
  
        return C_dst_SET, LOAD
        
    
    
    
     



   
     
    
     



