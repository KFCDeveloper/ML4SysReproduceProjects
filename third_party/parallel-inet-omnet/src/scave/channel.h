//=========================================================================
//  CHANNEL.H - part of
//                  OMNeT++/OMNEST
//           Discrete System Simulation in C++
//
//  Author: Andras Varga
//
//=========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 1992-2008 Andras Varga
  Copyright (C) 2006-2008 OpenSim Ltd.

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "commonutil.h"
#include <deque>
#include "node.h"

NAMESPACE_BEGIN


/**
 * Does buffering between two processing nodes (Node).
 *
 * @see Node, Port, Datum
 */
class SCAVE_API Channel
{
    private:
        // note: a Channel should *never* hold a pointer back to its Ports
        // because ports may be copied after having been assigned to channels
        // (e.g. in VectorFileReader which uses std::vector). Node ptrs are OK.
        std::deque<Datum> buffer;  //XXX deque has very poor performance under VC++ (pagesize==1!), consider using std::vector here instead
        Node *producernode;
        Node *consumernode;
        bool producerfinished;
        bool consumerfinished;
    public:
        Channel();
        ~Channel() {}

        void setProducerNode(Node *node) {producernode = node;}
        Node *getProducerNode() const {return producernode;}

        void setConsumerNode(Node *node) {consumernode = node;}
        Node *getConsumerNode() const {return consumernode;}

        /**
         * Returns ptr to the first buffered data item (next one to be read), or NULL
         */
        const Datum *peek() const;

        /**
         * Writes an array.
         */
        void write(Datum *a, int n);

        /**
         * Reads into an array. Returns number of items actually stored.
         */
        int read(Datum *a, int max);

        /**
         * Returns true if producer has already called close() which means
         * there will not be any more data except those already in the buffer
         */
        bool isClosing()  {return producerfinished;}

        /**
         * Returns true if close() has been called and there is no buffered data
         */
        bool eof()  {return producerfinished && length()==0;}

        /**
         * Called by the producer to declare it will not write any more --
         * if also there is no more buffered data (length()==0), that means EOF.
         */
        void close()  {producerfinished=true;}

        /**
         * Called when consumer has finished. Causes channel to ignore
         * further writes (discard any data written).
         */
        void consumerClose() {buffer.clear();consumerfinished=true;}

        /**
         * Returns true when the consumer has closed the channel, that is,
         * it will not read any more data from the channel.
         */
        bool isConsumerClosed() {return consumerfinished;}

        /**
         * Number of currently buffered items.
         */
        int length() {return buffer.size();}
};

NAMESPACE_END


#endif


