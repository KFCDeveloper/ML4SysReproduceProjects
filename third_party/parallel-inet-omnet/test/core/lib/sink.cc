#include <omnetpp.h>

namespace testlib {

class Sink : public cSimpleModule
{
  protected:
    int numPackets;
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
    virtual void finish();
};

Define_Module(Sink);

void Sink::initialize()
{
    ev << "Initializing Sink " << getFullPath() << "\n";
    numPackets = 0;
}

void Sink::handleMessage(cMessage *msg)
{
    ev << "Sink received msg " << msg->getName() << "\n";
    numPackets++;
    delete msg;
}

void Sink::finish()
{
    ev << "Sink finishing, received msgs: " << numPackets << "\n";
}

}

