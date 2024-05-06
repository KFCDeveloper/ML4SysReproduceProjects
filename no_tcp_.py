"""
A basic example that showcases how TCP can be used to generate packets, and how a TCP sink
can send acknowledgment packets back to the sender in a simple two-hop network.
"""

import simpy

from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire
from ns.switch.switch import SimpleDisPacketSwitch
from ns.packet.sink import PacketSink
import random
import numpy as np
from functools import partial
from ns.utils.generators.MAP_MSP_generator import BMAP_generator


def packet_arrival():
    """Packets arrive with a constant interval of 0.1 seconds."""
    return 0.008  # 0.1  0.0008


def packet_size():
    """The packets have a constant size of 1024 bytes."""
    return 1024  # 512


def delay_dist():
    """Network wires experience a constant propagation delay of 0.1 seconds."""
    return 0.1 # 0.1


def genfib_chain(tem_flow_num, tem_switch_port_num):
    tem_fib = {}
    for i in range(tem_flow_num):
        # tem_fib[i] = random.randint(0, tem_switch_port_num / 2 - 1)
        # tem_fib[i + 10000] = random.randint(tem_switch_port_num / 2, tem_switch_port_num - 1)
        tem_fib[i] = int(i % (tem_switch_port_num))

    return tem_fib


def interarrival(y):
    try:
        return next(y)
    except StopIteration:
        return

def const_size():
    """Constant packet size in bytes."""
    return 1024



def main():
    env = simpy.Environment()
    # (2) to generate inter-arrival times ~ MAP or BMAP model
    D0 = np.array([[-114.46031, 11.3081, 8.42701],
                   [158.689, -29152.1587, 20.5697],
                   [1.08335, 0.188837, -1.94212]])
    D1 = np.array([[94.7252, 0.0, 0.0], [0.0, 2.89729e4, 0.0],
                   [0.0, 0.0, 0.669933]])
    y = BMAP_generator([D0, D1])

    iat_dist = partial(interarrival, y)
    # pkt_size_dist = partial(packet_size, myseed=10)
    pkt_size_dist = partial(packet_size)
    
    # set flow
    flow_num = 4  # 1
    all_flows = []
    for i in range(flow_num):
        pg = DistPacketGenerator(
            env,
            "flow_"+str(i),
            iat_dist,
            const_size,
            initial_delay=0.0,
            finish=10,
            flow_id=i,
            rec_flow=True,
        )
        # each_flow = Flow(
        #     fid=i,
        #     src="flow " + str(i),
        #     dst="flow " + str(i),
        #     finish_time=10,
        #     arrival_dist=packet_arrival,
        #     size_dist=packet_size, )
        all_flows.append(pg)

    # set switch: switches arrange in a chain
    switch_num = 1
    switch_port_num = 4
    switch_buffer_size = 512*8*8*8*8
    switch_port_rate = 16384*8*8*8

    switch = SimpleDisPacketSwitch(
        env, pkt_size_dist,
        nports=switch_port_num,
        port_rate=switch_port_rate,  # in bits/second
        buffer_size=switch_buffer_size,  # in packets
        debug=True,
    )

    ps = PacketSink(env)
    for i in range(switch_port_num):
        switch.ports[i].out = ps
    # switch = FairPacketSwitch(
    #     env,
    #     nports=1,
    #     port_rate=port_rate,
    #     buffer_size=buffer_size,
    #     weights=[1, 2],
    #     server="DRR",
    #     debug=True,
    # )

    # I find that if I did not model link and only use one switch. We don't need to do anything with Port; 
    # but dataset I need to record the 
    fib = genfib_chain(flow_num, switch_port_num)  # fixed this, make it convenient for debugging
    # fib = {0: 1, 10000: 3, 1: 0, 10001: 2, 2: 1, 10002: 2, 3: 0, 10003: 2, 4: 1, 10004: 3, 5: 1, 10005: 2, 6: 1, 10006: 2, 7: 1, 10007: 2}
    switch.demux.fib = fib

    for flow_index in range(len(all_flows)):
        sender = all_flows[flow_index]
        sender.out = switch
        # if not switch.ports[fib[sender.flow_id]]:   # 这里的模型是，入口不会有queue，全部都在port对应的queue里；所以在create switch的时候，把所有port都创建一个sink也没问题
        #     switch.ports[fib[sender.flow_id]].out = 

    env.run(until=100)


main()
