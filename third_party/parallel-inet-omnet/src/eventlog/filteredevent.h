//=========================================================================
//  FILTEREDEVENT.H - part of
//                  OMNeT++/OMNEST
//           Discrete System Simulation in C++
//
//  Author: Levente Meszaros
//
//=========================================================================

/*--------------------------------------------------------------*
  Copyright (C) 2006-2008 OpenSim Ltd.

  This file is distributed WITHOUT ANY WARRANTY. See the file
  `license' for details on this and other legal matters.
*--------------------------------------------------------------*/

#ifndef __FILTEREDEVENT_H_
#define __FILTEREDEVENT_H_

#include <vector>
#include "eventlogdefs.h"
#include "ievent.h"
#include "event.h"
#include "messagedependency.h"

NAMESPACE_BEGIN

class FilteredEventLog;

/**
 * Events stored in the FilteredEventLog. This class uses the Event class by delegation so that multiple
 * filtered events can share the same event object.
 *
 * Filtered events are in a lazy double-linked list based on event numbers.
 */
class EVENTLOG_API FilteredEvent : public IEvent
{
    protected:
        FilteredEventLog *filteredEventLog;

        eventnumber_t eventNumber; // the corresponding event number
        eventnumber_t causeEventNumber; // the event number from which the message was sent that is being processed in this event
        IMessageDependency *cause; // the message send which is processed in this event
        IMessageDependencyList *causes; // the arrival message sends of messages which we send in this even and are in the filtered set
        IMessageDependencyList *consequences; // the message sends and arrivals from this event to another in the filtered set

        struct BreadthSearchItem {
            IEvent *event;
            IMessageDependency *firstSeenMessageDependency;
            FilteredMessageDependency::Kind effectiveKind;
            int level;

            BreadthSearchItem(IEvent *event, IMessageDependency *firstSeenMessageDependency, FilteredMessageDependency::Kind effectiveKind, int level)
            {
                this->event = event;
                this->firstSeenMessageDependency = firstSeenMessageDependency;
                this->effectiveKind = effectiveKind;
                this->level = level;
            }
        };

    public:
        FilteredEvent(FilteredEventLog *filteredEventLog, eventnumber_t eventNumber);
        virtual ~FilteredEvent();

        IEvent *getEvent();

        // IEvent interface
        virtual void synchronize(FileReader::FileChangedState change);
        virtual IEventLog *getEventLog();

        virtual ModuleCreatedEntry *getModuleCreatedEntry() { return getEvent()->getModuleCreatedEntry(); }

        virtual file_offset_t getBeginOffset() { return getEvent()->getBeginOffset(); }
        virtual file_offset_t getEndOffset() { return getEvent()->getEndOffset(); }

        virtual EventEntry *getEventEntry() { return getEvent()->getEventEntry(); }
        virtual int getNumEventLogEntries() { return getEvent()->getNumEventLogEntries(); }
        virtual EventLogEntry *getEventLogEntry(int index) { return getEvent()->getEventLogEntry(index); }

        virtual int getNumEventLogMessages() { return getEvent()->getNumEventLogMessages(); }
        virtual int getNumBeginSendEntries() { return getEvent()->getNumBeginSendEntries(); }
        virtual EventLogMessageEntry *getEventLogMessage(int index) { return getEvent()->getEventLogMessage(index); }

        virtual eventnumber_t getEventNumber() { return eventNumber; }
        virtual simtime_t getSimulationTime() { return getEvent()->getSimulationTime(); }
        virtual int getModuleId() { return getEvent()->getModuleId(); }
        virtual long getMessageId() { return getEvent()->getMessageId(); }
        virtual eventnumber_t getCauseEventNumber() { return getEvent()->getCauseEventNumber(); }

        virtual bool isSelfMessage(BeginSendEntry *beginSendEntry) { return getEvent()->isSelfMessage(beginSendEntry); }
        virtual bool isSelfMessageProcessingEvent() { return getEvent()->isSelfMessageProcessingEvent(); }
        virtual EndSendEntry *getEndSendEntry(BeginSendEntry *beginSendEntry) { return getEvent()->getEndSendEntry(beginSendEntry); };
        virtual FilteredEvent *getPreviousEvent();
        virtual FilteredEvent *getNextEvent();

        virtual FilteredEvent *getCauseEvent();
        virtual IMessageDependency *getCause();
        virtual BeginSendEntry *getCauseBeginSendEntry();
        virtual IMessageDependencyList *getCauses();
        virtual IMessageDependencyList *getConsequences();

        virtual void print(FILE *file = stdout, bool outputEventLogMessages = true) { getEvent()->print(file, outputEventLogMessages); }

    protected:
        void clearInternalState();
        void deleteConsequences();
        void deleteAllocatedObjects();

        void getCauses(IEvent *event, IMessageDependency *endMessageDependency, bool IsMixed, int level);
        void getConsequences(IEvent *event, IMessageDependency *beginMessageDependency, bool IsMixed, int level);
        int countFilteredMessageDependencies(IMessageDependencyList *messageDependencies);
        void pushNewFilteredMessageDependency(IMessageDependencyList *messageDependencies, FilteredMessageDependency::Kind kind, IMessageDependency *beginMessageDependency, IMessageDependency *endMessageDependency);
};

NAMESPACE_END


#endif
