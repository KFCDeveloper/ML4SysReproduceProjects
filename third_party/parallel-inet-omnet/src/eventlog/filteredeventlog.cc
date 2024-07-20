//=========================================================================
//  FILTEREDEVENTLOG.CC - part of
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

#include <stdio.h>
#include <algorithm>
#include "filteredeventlog.h"

NAMESPACE_BEGIN

FilteredEventLog::FilteredEventLog(IEventLog *eventLog)
{
    this->eventLog = eventLog;
    // collection limit parameters
    collectMessageReuses = true;
    maximumNumberOfCauses = maximumNumberOfConsequences = 5;
    maximumCauseDepth = maximumConsequenceDepth = 15;
    maximumCauseCollectionTime = maximumConsequenceCollectionTime = 100;
    // trace filter parameters
    tracedEventNumber = -1;
    firstEventNumber = -1;
    lastEventNumber = -1;
    traceCauses = true;
    traceConsequences = true;
    traceMessageReuses = true;
    traceSelfMessages = true;
    // other filter parameters
    enableModuleFilter = false;
    enableMessageFilter = false;
    setModuleExpression("");
    setMessageExpression("");
    clearInternalState();
}

FilteredEventLog::~FilteredEventLog()
{
    deleteAllocatedObjects();
}

void FilteredEventLog::clearInternalState()
{
    firstMatchingEvent = NULL;
    lastMatchingEvent = NULL;
    approximateNumberOfEvents = -1;
    approximateMatchingEventRatio = -1;
    eventNumberToFilteredEventMap.clear();
    eventNumberToFilterMatchesFlagMap.clear();
    eventNumberToTraceableEventFlagMap.clear();
    unseenTracedEventCauseEventNumbers.clear();
    unseenTracedEventConsequenceEventNumbers.clear();
}

void FilteredEventLog::deleteAllocatedObjects()
{
    for (EventNumberToFilteredEventMap::iterator it = eventNumberToFilteredEventMap.begin(); it != eventNumberToFilteredEventMap.end(); it++)
        delete it->second;
    eventNumberToFilteredEventMap.clear();
    eventNumberToFilterMatchesFlagMap.clear();
    eventNumberToTraceableEventFlagMap.clear();
}

void FilteredEventLog::synchronize(FileReader::FileChangedState change)
{
    if (change != FileReader::UNCHANGED) {
        eventLog->synchronize(change);
        switch (change) {
            case FileReader::UNCHANGED:   // just to avoid unused enumeration value warnings
                 break;
            case FileReader::OVERWRITTEN:
                deleteAllocatedObjects();
                clearInternalState();
                break;
            case FileReader::APPENDED:
                for (EventNumberToFilteredEventMap::iterator it = eventNumberToFilteredEventMap.begin(); it != eventNumberToFilteredEventMap.end(); it++)
                    it->second->synchronize(change);
                if (lastMatchingEvent) {
                    eventnumber_t eventNumber = lastMatchingEvent->getEventNumber();
                    eventNumberToFilteredEventMap.erase(eventNumber);
                    eventNumberToFilterMatchesFlagMap.erase(eventNumber);
                    eventNumberToTraceableEventFlagMap.erase(eventNumber);
                    if (firstMatchingEvent == lastMatchingEvent)
                        firstMatchingEvent = NULL;
                    delete lastMatchingEvent;
                    lastMatchingEvent = NULL;
                }
                break;
        }
    }
}

void FilteredEventLog::setPatternMatchers(std::vector<PatternMatcher> &patternMatchers, std::vector<std::string> &patterns, bool dottedPath)
{
    for (std::vector<std::string>::iterator it = patterns.begin(); it != patterns.end(); it++) {
        PatternMatcher matcher;
        matcher.setPattern((*it).c_str(), dottedPath, true, false);
        patternMatchers.push_back(matcher);
    }
}

eventnumber_t FilteredEventLog::getApproximateNumberOfEvents()
{
    if (approximateNumberOfEvents == -1)
    {
        if (tracedEventNumber != -1) {
            // TODO: this is clearly not good and should return a much better approximation
            // TODO: maybe start from traced event number and go forward/backward and return approximation based on that?
            if (firstEventNumber != -1 && lastEventNumber != -1)
                return lastEventNumber - firstEventNumber;
            else
                return 1000;
        }
        else {
            // TODO: what if filter is event range limited?
            FilteredEvent *firstEvent = getFirstEvent();
            FilteredEvent *lastEvent = getLastEvent();

            if (firstEvent == NULL)
                approximateNumberOfEvents = 0;
            else
            {
                file_offset_t beginOffset = firstEvent->getBeginOffset();
                file_offset_t endOffset = lastEvent->getEndOffset();
                long sumMatching = 0;
                long sumNotMatching = 0;
                long count = 0;
                int eventCount = 100;

                // TODO: perhaps it would be better to read in random events
                for (int i = 0; i < eventCount; i++)
                {
                    if (firstEvent) {
                        FilteredEvent *previousEvent = firstEvent;
                        sumMatching += firstEvent->getEndOffset() - firstEvent->getBeginOffset();
                        count++;
                        firstEvent = firstEvent->getNextEvent();
                        if (firstEvent)
                            sumNotMatching += firstEvent->getBeginOffset() - previousEvent->getEndOffset();
                    }

                    if (lastEvent) {
                        FilteredEvent *previousEvent = lastEvent;
                        sumMatching += lastEvent->getEndOffset() - lastEvent->getBeginOffset();
                        count++;
                        lastEvent = lastEvent->getPreviousEvent();
                        if (lastEvent)
                            sumNotMatching += previousEvent->getBeginOffset() - lastEvent->getEndOffset();
                    }
                }

                double averageMatching = (double)sumMatching / count;
                double averageNotMatching = (double)sumNotMatching / count;
                approximateMatchingEventRatio = averageMatching / (averageMatching + averageNotMatching);
                approximateNumberOfEvents = (long)((endOffset - beginOffset) / averageMatching * approximateMatchingEventRatio);
            }
        }
    }

    return approximateNumberOfEvents;
}

double FilteredEventLog::getApproximatePercentageForEventNumber(eventnumber_t eventNumber)
{
    if (tracedEventNumber != -1)
        // TODO: this is clearly not good and should return a much better approximation
        return IEventLog::getApproximatePercentageForEventNumber(eventNumber);
    else
        // TODO: what if filter is event range limited
        return IEventLog::getApproximatePercentageForEventNumber(eventNumber);
}

FilteredEvent *FilteredEventLog::getApproximateEventAt(double percentage)
{
    if (isEmpty())
        return NULL;
    else {
        double firstEventPercentage = eventLog->getApproximatePercentageForEventNumber(getFirstEvent()->getEventNumber());
        double lastEventPercentage = eventLog->getApproximatePercentageForEventNumber(getLastEvent()->getEventNumber());
        percentage = firstEventPercentage + percentage * (lastEventPercentage - firstEventPercentage);
        IEvent *event = eventLog->getApproximateEventAt(percentage);

        FilteredEvent *filteredEvent = getMatchingEventInDirection(event, true);
        if (filteredEvent)
            return filteredEvent;

        filteredEvent = getMatchingEventInDirection(event, false);
        if (filteredEvent)
            return filteredEvent;

        Assert(false);
        return NULL;
    }
}

FilteredEvent *FilteredEventLog::getNeighbourEvent(IEvent *event, eventnumber_t distance)
{
    return (FilteredEvent *)IEventLog::getNeighbourEvent(event, distance);
}

bool FilteredEventLog::matchesFilter(IEvent *event)
{
    Assert(event);
    EventNumberToBooleanMap::iterator it = eventNumberToFilterMatchesFlagMap.find(event->getEventNumber());

    // if cached, return it
    if (it != eventNumberToFilterMatchesFlagMap.end())
        return it->second;

    //printf("*** Matching filter to event: %ld\n", event->getEventNumber());

    bool matches = matchesEvent(event) && matchesDependency(event);
    eventNumberToFilterMatchesFlagMap[event->getEventNumber()] = matches;
    return matches;
}

bool FilteredEventLog::matchesEvent(IEvent *event)
{
    // event outside of considered range
    if ((firstEventNumber != -1 && event->getEventNumber() < firstEventNumber) ||
        (lastEventNumber != -1 && event->getEventNumber() > lastEventNumber))
        return false;

    // event's module
    if (enableModuleFilter) {
        ModuleCreatedEntry *eventModuleCreatedEntry = event->getModuleCreatedEntry();
        ModuleCreatedEntry *moduleCreatedEntry = eventModuleCreatedEntry;
        // match parent chain of event's module (to handle compound modules too)
        while (moduleCreatedEntry) {
            if (matchesModuleCreatedEntry(moduleCreatedEntry)) {
                if (moduleCreatedEntry == eventModuleCreatedEntry)
                    goto MATCHES;
                else {
                    // check if the event has a cause or consequence referring
                    // outside the matching compound module
                    IMessageDependencyList *causes = event->getCauses();
                    for (IMessageDependencyList::iterator it = causes->begin(); it != causes->end(); it++) {
                        IEvent *causeEvent = (*it)->getCauseEvent();
                        if (causeEvent && !isAncestorModuleCreatedEntry(moduleCreatedEntry, causeEvent->getModuleCreatedEntry()))
                            goto MATCHES;
                    }

                    IMessageDependencyList *consequences = event->getConsequences();
                    for (IMessageDependencyList::iterator it = consequences->begin(); it != consequences->end(); it++) {
                        IEvent *consequenceEvent = (*it)->getConsequenceEvent();
                        if (consequenceEvent && !isAncestorModuleCreatedEntry(moduleCreatedEntry, consequenceEvent->getModuleCreatedEntry()))
                            goto MATCHES;
                    }
                }
            }

            moduleCreatedEntry = getModuleCreatedEntry(moduleCreatedEntry->parentModuleId);
        }

        // no match
        return false;
        // match found
        MATCHES:;
    }

    // event's message
    if (enableMessageFilter) {
        BeginSendEntry *beginSendEntry = event->getCauseBeginSendEntry();
        bool matches = beginSendEntry ? matchesBeginSendEntry(beginSendEntry) : false;

        for (int i = 0; i < event->getNumEventLogEntries(); i++) {
            beginSendEntry = dynamic_cast<BeginSendEntry *>(event->getEventLogEntry(i));

            if (beginSendEntry && matchesBeginSendEntry(beginSendEntry)) {
                matches = true;
                break;
            }
        }

        if (!matches)
            return false;
    }

    return true;
}

bool FilteredEventLog::matchesDependency(IEvent *event)
{
    // if there's no traced event
    if (tracedEventNumber == -1)
        return true;

    // if this is the traced event
    if (event->getEventNumber() == tracedEventNumber)
        return true;

    // event is cause of traced event
    if (tracedEventNumber > event->getEventNumber() && traceCauses)
        return isCauseOfTracedEvent(event);

    // event is consequence of traced event
    if (tracedEventNumber < event->getEventNumber() && traceConsequences)
        return isConsequenceOfTracedEvent(event);

    return false;
}

bool FilteredEventLog::matchesModuleCreatedEntry(ModuleCreatedEntry *moduleCreatedEntry)
{
    return
        matchesExpression(moduleExpression, moduleCreatedEntry) ||
        matchesPatterns(moduleNames, moduleCreatedEntry->fullName) ||
        matchesPatterns(moduleClassNames, moduleCreatedEntry->moduleClassName) ||
        matchesPatterns(moduleNEDTypeNames, moduleCreatedEntry->nedTypeName) ||
        matchesList(moduleIds, moduleCreatedEntry->moduleId);
}

bool FilteredEventLog::matchesBeginSendEntry(BeginSendEntry *beginSendEntry)
{
    return
        matchesExpression(messageExpression, beginSendEntry) ||
        matchesPatterns(messageNames, beginSendEntry->messageName) ||
        matchesPatterns(messageClassNames, beginSendEntry->messageClassName) ||
        matchesList(messageIds, beginSendEntry->messageId) ||
        matchesList(messageTreeIds, beginSendEntry->messageTreeId) ||
        matchesList(messageEncapsulationIds, beginSendEntry->messageEncapsulationId) ||
        matchesList(messageEncapsulationTreeIds, beginSendEntry->messageEncapsulationTreeId);
}

bool FilteredEventLog::matchesExpression(MatchExpression &matchExpression, EventLogEntry *eventLogEntry)
{
    return matchExpression.matches(eventLogEntry);
}

bool FilteredEventLog::matchesPatterns(std::vector<PatternMatcher> &patterns, const char *str)
{
    if (patterns.empty())
        return false;

    for (std::vector<PatternMatcher>::iterator it = patterns.begin(); it != patterns.end(); it++)
        if ((*it).matches(str))
            return true;

    return false;
}

template <typename T> bool FilteredEventLog::matchesList(std::vector<T> &elements, T element)
{
    if (elements.empty())
        return false;
    else
        return std::find(elements.begin(), elements.end(), element) != elements.end();
}

bool FilteredEventLog::isEmpty()
{
    if (tracedEventNumber != -1) {
        IEvent *event = eventLog->getEventForEventNumber(tracedEventNumber);

        if (event && matchesFilter(event))
            return false;
    }

    return IEventLog::isEmpty();
}

FilteredEvent *FilteredEventLog::getFirstEvent()
{
    if (!firstMatchingEvent && !eventLog->isEmpty())
    {
        eventnumber_t startEventNumber = firstEventNumber == -1 ? eventLog->getFirstEvent()->getEventNumber() : std::max(eventLog->getFirstEvent()->getEventNumber(), firstEventNumber);
        firstMatchingEvent = getMatchingEventInDirection(startEventNumber, true);
    }

    return firstMatchingEvent;
}

FilteredEvent *FilteredEventLog::getLastEvent()
{
    if (!lastMatchingEvent && !eventLog->isEmpty())
    {
        eventnumber_t startEventNumber = lastEventNumber == -1 ? eventLog->getLastEvent()->getEventNumber() : std::min(eventLog->getLastEvent()->getEventNumber(), lastEventNumber);
        lastMatchingEvent = getMatchingEventInDirection(startEventNumber, false);
    }

    return lastMatchingEvent;
}

FilteredEvent *FilteredEventLog::getEventForEventNumber(eventnumber_t eventNumber, MatchKind matchKind, bool useCacheOnly)
{
    Assert(eventNumber >= 0);
    EventNumberToFilteredEventMap::iterator it = eventNumberToFilteredEventMap.find(eventNumber);
    if (it != eventNumberToFilteredEventMap.end())
        return it->second;
    IEvent *event = eventLog->getEventForEventNumber(eventNumber, matchKind, useCacheOnly);
    if (event) {
        switch (matchKind) {
            case EXACT:
                if (matchesFilter(event))
                    return cacheFilteredEvent(event->getEventNumber());
                break;
            case FIRST_OR_PREVIOUS:
                if (event->getEventNumber() == eventNumber && matchesFilter(event))
                    return cacheFilteredEvent(event->getEventNumber());
                else if (!useCacheOnly)
                    return getMatchingEventInDirection(event, false);
            case FIRST_OR_NEXT:
                if (!useCacheOnly)
                    return getMatchingEventInDirection(event, true);
            case LAST_OR_PREVIOUS:
                if (!useCacheOnly)
                    return getMatchingEventInDirection(event, false);
            case LAST_OR_NEXT:
                if (event->getEventNumber() == eventNumber && matchesFilter(event))
                    return cacheFilteredEvent(event->getEventNumber());
                else if (!useCacheOnly)
                    return getMatchingEventInDirection(event, true);
        }
    }
    return NULL;
}

FilteredEvent *FilteredEventLog::getEventForSimulationTime(simtime_t simulationTime, MatchKind matchKind, bool useCacheOnly)
{
    IEvent *event = eventLog->getEventForSimulationTime(simulationTime, matchKind, useCacheOnly);
    if (event) {
        switch (matchKind) {
            case EXACT:
                if (matchesFilter(event))
                    return cacheFilteredEvent(event->getEventNumber());
                break;
            case FIRST_OR_PREVIOUS:
                if (!useCacheOnly) {
                    if (event->getSimulationTime() == simulationTime) {
                        IEvent *lastEvent = eventLog->getEventForSimulationTime(simulationTime, LAST_OR_NEXT);
                        FilteredEvent *matchingEvent = getMatchingEventInDirection(event, true, lastEvent->getEventNumber());

                        if (matchingEvent)
                            return matchingEvent;
                    }
                    return getMatchingEventInDirection(event, false);
                }
            case FIRST_OR_NEXT:
                if (!useCacheOnly)
                    return getMatchingEventInDirection(event, true);
            case LAST_OR_PREVIOUS:
                if (!useCacheOnly)
                    return getMatchingEventInDirection(event, false);
            case LAST_OR_NEXT:
                if (!useCacheOnly) {
                    if (event->getSimulationTime() == simulationTime) {
                        IEvent *firstEvent = eventLog->getEventForSimulationTime(simulationTime, FIRST_OR_PREVIOUS);
                        FilteredEvent *matchingEvent = getMatchingEventInDirection(event, false, firstEvent->getEventNumber());

                        if (matchingEvent)
                            return matchingEvent;
                    }
                    return getMatchingEventInDirection(event, true);
                }
        }
    }

    return NULL;
}

EventLogEntry *FilteredEventLog::findEventLogEntry(EventLogEntry *start, const char *search, bool forward, bool caseSensitive)
{
    EventLogEntry *eventLogEntry = start;

    do {
        eventLogEntry = eventLog->findEventLogEntry(eventLogEntry, search, forward, caseSensitive);
    }
    while (eventLogEntry && !matchesFilter(eventLogEntry->getEvent()));

    return eventLogEntry;
}

FilteredEvent *FilteredEventLog::getMatchingEventInDirection(eventnumber_t eventNumber, bool forward, eventnumber_t stopEventNumber)
{
    Assert(eventNumber >= 0);
    IEvent *event = eventLog->getEventForEventNumber(eventNumber);

    return getMatchingEventInDirection(event, forward, stopEventNumber);
}

FilteredEvent *FilteredEventLog::getMatchingEventInDirection(IEvent *event, bool forward, eventnumber_t stopEventNumber)
{
    Assert(event);

    // optimization
    if (forward) {
        if (firstMatchingEvent && event->getEventNumber() < firstMatchingEvent->getEventNumber()) {
            if (stopEventNumber != -1 && stopEventNumber < firstMatchingEvent->getEventNumber())
                return NULL;
            else
                return firstMatchingEvent;
        }

        if (firstEventNumber != -1 && event->getEventNumber() < firstEventNumber)
            event = eventLog->getEventForEventNumber(firstEventNumber, LAST_OR_NEXT);
    }
    else {
        if (lastMatchingEvent && lastMatchingEvent->getEventNumber() < event->getEventNumber()) {
            if (stopEventNumber != -1 && lastMatchingEvent->getEventNumber() < stopEventNumber)
                return NULL;
            else
                return lastMatchingEvent;
        }
        if (lastEventNumber != -1 && lastEventNumber < event->getEventNumber())
            event = eventLog->getEventForEventNumber(lastEventNumber, FIRST_OR_PREVIOUS);
    }

    Assert(event);

    // TODO: LONG RUNNING OPERATION
    // if none of firstEventNumber, lastEventNumber, stopEventNumber is set this might take a while
    while (event)
    {
        eventLog->progress();
        int eventNumber = event->getEventNumber();

        if (matchesFilter(event))
            return cacheFilteredEvent(eventNumber);

        if (forward)
        {
            eventNumber++;
            event = event->getNextEvent();

            if (lastEventNumber != -1 && eventNumber > lastEventNumber)
                return NULL;

            if (stopEventNumber != -1 && eventNumber > stopEventNumber)
                return NULL;
        }
        else {
            eventNumber--;
            event = event->getPreviousEvent();

            if (firstEventNumber != -1 && eventNumber < firstEventNumber)
                return NULL;

            if (stopEventNumber != -1 && eventNumber < stopEventNumber)
                return NULL;
        }
    }

    return NULL;
}

void FilteredEventLog::setTracedEventNumber(eventnumber_t tracedEventNumber)
{
    Assert(this->tracedEventNumber == -1);
    this->tracedEventNumber = tracedEventNumber;
    unseenTracedEventCauseEventNumbers.push_back(tracedEventNumber);
    unseenTracedEventConsequenceEventNumbers.push_back(tracedEventNumber);
}

// TODO: LONG RUNNING OPERATION
// this does a recursive depth search
bool FilteredEventLog::isCauseOfTracedEvent(IEvent *causeEvent)
{
    Assert(causeEvent);
    eventLog->progress();
    EventNumberToBooleanMap::iterator it = eventNumberToTraceableEventFlagMap.find(causeEvent->getEventNumber());
    if (it != eventNumberToTraceableEventFlagMap.end())
        return it->second;
    //printf("Checking if %ld is cause of %ld\n", causeEvent->getEventNumber(), tracedEventNumber);

    eventnumber_t causeEventNumber = causeEvent->getEventNumber();
    while (!unseenTracedEventCauseEventNumbers.empty() && unseenTracedEventCauseEventNumbers.front() >= causeEventNumber) {
        eventnumber_t unseenTracedEventCauseEventNumber = unseenTracedEventCauseEventNumbers.front();
        unseenTracedEventCauseEventNumbers.pop_front();
        IEvent *unseenTracedEventCauseEvent = eventLog->getEventForEventNumber(unseenTracedEventCauseEventNumber);
        if (unseenTracedEventCauseEvent) {
            IMessageDependencyList *causes = unseenTracedEventCauseEvent->getCauses();
            for (IMessageDependencyList::iterator it = causes->begin(); it != causes->end(); it++)
            {
                IMessageDependency *messageDependency = *it;
                IEvent *newUnseenTracedEventCauseEvent = messageDependency->getCauseEvent();
                if (newUnseenTracedEventCauseEvent &&
                    (traceSelfMessages || !newUnseenTracedEventCauseEvent->isSelfMessageProcessingEvent()) &&
                    (traceMessageReuses || !dynamic_cast<MessageReuseDependency *>(messageDependency)))
                {
                    eventnumber_t newUnseenTracedEventCauseEventNumber = newUnseenTracedEventCauseEvent->getEventNumber();
                    eventNumberToTraceableEventFlagMap[newUnseenTracedEventCauseEventNumber] = true;
                    unseenTracedEventCauseEventNumbers.push_back(newUnseenTracedEventCauseEventNumber);
                    if (newUnseenTracedEventCauseEventNumber == causeEventNumber)
                        return true;
                }
            }
            // TODO: this is far from being optimal, inserting the items in the right place would be more desirable
            sort(unseenTracedEventCauseEventNumbers.begin(), unseenTracedEventCauseEventNumbers.end());
        }
    }

    return eventNumberToTraceableEventFlagMap[causeEventNumber] = false;
}

// TODO: LONG RUNNING OPERATION
// this does a recursive depth search
bool FilteredEventLog::isConsequenceOfTracedEvent(IEvent *consequenceEvent)
{
    Assert(consequenceEvent);
    eventLog->progress();
    EventNumberToBooleanMap::iterator it = eventNumberToTraceableEventFlagMap.find(consequenceEvent->getEventNumber());
    if (it != eventNumberToTraceableEventFlagMap.end())
        return it->second;
    //printf("Checking if %ld is consequence of %ld\n", consequence->getEventNumber(), tracedEventNumber);

    // like isCauseOfTracedEvent(), but searching from the opposite direction
    eventnumber_t consequenceEventNumber = consequenceEvent->getEventNumber();
    while (!unseenTracedEventConsequenceEventNumbers.empty() && unseenTracedEventConsequenceEventNumbers.front() <= consequenceEventNumber) {
        eventnumber_t unseenTracedEventConsequenceEventNumber = unseenTracedEventConsequenceEventNumbers.front();
        unseenTracedEventConsequenceEventNumbers.pop_front();
        IEvent *unseenTracedEventConsequenceEvent = eventLog->getEventForEventNumber(unseenTracedEventConsequenceEventNumber);
        if (unseenTracedEventConsequenceEvent) {
            IMessageDependencyList *consequences = unseenTracedEventConsequenceEvent->getConsequences();
            for (IMessageDependencyList::iterator it = consequences->begin(); it != consequences->end(); it++)
            {
                IMessageDependency *messageDependency = *it;
                IEvent *newUnseenTracedEventConsequenceEvent = messageDependency->getConsequenceEvent();
                if (newUnseenTracedEventConsequenceEvent &&
                    (traceSelfMessages || !newUnseenTracedEventConsequenceEvent->isSelfMessageProcessingEvent()) &&
                    (traceMessageReuses || !dynamic_cast<MessageReuseDependency *>(messageDependency)))
                {
                    eventnumber_t newUnseenTracedEventConsequenceEventNumber = newUnseenTracedEventConsequenceEvent->getEventNumber();
                    eventNumberToTraceableEventFlagMap[newUnseenTracedEventConsequenceEventNumber] = true;
                    unseenTracedEventConsequenceEventNumbers.push_back(newUnseenTracedEventConsequenceEventNumber);
                    if (newUnseenTracedEventConsequenceEventNumber == consequenceEventNumber)
                        return true;
                }
            }
            // TODO: this is far from being optimal, inserting the items in the right place would be more desirable
            sort(unseenTracedEventConsequenceEventNumbers.begin(), unseenTracedEventConsequenceEventNumbers.end());
        }
    }

    return eventNumberToTraceableEventFlagMap[consequenceEventNumber] = false;
}

FilteredEvent *FilteredEventLog::cacheFilteredEvent(eventnumber_t eventNumber)
{
    EventNumberToFilteredEventMap::iterator it = eventNumberToFilteredEventMap.find(eventNumber);

    if (it != eventNumberToFilteredEventMap.end())
        return it->second;
    else {
        FilteredEvent *filteredEvent = new FilteredEvent(this, eventNumber);
        eventNumberToFilteredEventMap[eventNumber] = filteredEvent;
        return filteredEvent;
    }
}

bool FilteredEventLog::isAncestorModuleCreatedEntry(ModuleCreatedEntry *ancestor, ModuleCreatedEntry *descendant)
{
    while (descendant) {
        if (descendant == ancestor)
            return true;
        else
            descendant = getModuleCreatedEntry(descendant->parentModuleId);
    }

    return false;
}

NAMESPACE_END

