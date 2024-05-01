"""
A basic example that showcases how TCP can be used to generate packets, and how a TCP sink
can send acknowledgment packets back to the sender in a simple two-hop network.
"""

import simpy

from ns.flow.cc import TCPReno
from ns.flow.cubic import TCPCubic
from ns.flow.flow import AppType, Flow
from ns.packet.tcp_generator import TCPPacketGenerator
from ns.packet.tcp_sink import TCPSink
from ns.port.wire import Wire
from ns.switch.switch import SimpleDisPacketSwitch
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
        tem_fib[i] = random.randint(0, tem_switch_port_num / 2 - 1)
        tem_fib[i + 10000] = random.randint(tem_switch_port_num / 2, tem_switch_port_num - 1)
    return tem_fib


def interarrival(y):
    try:
        return next(y)
    except StopIteration:
        return


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
        each_flow = Flow(
            fid=i,
            src="flow " + str(i),
            dst="flow " + str(i),
            finish_time=10,
            arrival_dist=packet_arrival,
            size_dist=packet_size, )
        all_flows.append(each_flow)

    # set switch: switches arrange in a chain
    switch_num = 1
    switch_port_num = 4
    switch_buffer_size = 5
    switch_port_rate = 16384

    switch = SimpleDisPacketSwitch(
        env, pkt_size_dist,
        nports=switch_port_num,
        port_rate=switch_port_rate,  # in bits/second
        buffer_size=switch_buffer_size,  # in packets
        debug=True,
    )
    # I find that if I did not model link and only use one switch. We don't need to do anything with Port; 
    # but dataset I need to record the 
    fib = genfib_chain(flow_num, switch_port_num)  # fixed this, make it convenient for debugging
    # fib = {0: 1, 10000: 3, 1: 0, 10001: 2, 2: 1, 10002: 2, 3: 0, 10003: 2, 4: 1, 10004: 3, 5: 1, 10005: 2, 6: 1, 10006: 2, 7: 1, 10007: 2}
    switch.demux.fib = fib

    for flow_index in range(len(all_flows)):
        sender = TCPPacketGenerator(env, flow=all_flows[flow_index], cc=TCPReno(),
                                    element_id="sender_" + str(flow_index), debug=True)
        receiver = TCPSink(env, rec_waits=True, debug=True)

        this_flow = all_flows[flow_index]

        this_flow.pkt_gen = sender
        this_flow.pkt_sink = receiver
        this_flow.pkt_gen.out = switch
        switch.demux.outs[fib[flow_index]].out = this_flow.pkt_sink

        receiver.out = switch  # ack flow
        switch.demux.outs[fib[flow_index + 10000]].out = this_flow.pkt_gen  # ack flow
        # ft.nodes[flow.dst]["device"].demux.ends[flow_id] 使用ends不行，但是使用 outs 上面这样写，却可以
        # 还没弄清楚 outs 和 ends 之间的关系

    # wire1_downstream = Wire(env, delay_dist)
    # wire1_upstream = Wire(env, delay_dist)
    # wire2_downstream = Wire(env, delay_dist)
    # wire2_upstream = Wire(env, delay_dist)

    # switch = SimplePacketSwitch(
    #     env,
    #     nports=2,
    #     port_rate=16384,  # in bits/second
    #     buffer_size=5,  # in packets
    #     debug=True,
    # )

    # receiver = TCPSink(env, rec_waits=True, debug=True)

    # # Sender-wire1-Switch-wire2-Reciever
    # # each wire has downstream and upstream
    # # switch has two port, and each port connet with a wire.
    # sender.out = wire1_downstream
    # wire1_downstream.out = switch
    # wire2_downstream.out = receiver
    # receiver.out = wire2_upstream
    # wire2_upstream.out = switch

    # fib = {0: 0, 10000: 1}  # Flow Information Base 
    # switch.demux.fib = fib
    # switch.demux.outs[0].out = wire2_downstream
    # switch.demux.outs[1].out = wire1_upstream

    # wire1_upstream.out = sender

    env.run(until=100)


main()
