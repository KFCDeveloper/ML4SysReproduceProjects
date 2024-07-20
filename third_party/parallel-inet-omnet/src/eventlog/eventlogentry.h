//=========================================================================
//  EVENTLOGENTRY.H - part of
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

#ifndef __EVENTLOGENTRY_H_
#define __EVENTLOGENTRY_H_

#include <sstream>
#include "platmisc.h"
#include "matchexpression.h"
#include "eventlogdefs.h"
#include "linetokenizer.h"

NAMESPACE_BEGIN

class Event;
class EventLog;

/**
 * Base class for all kind of event log entries.
 * An entry is represented by a single line in the log file.
 */
class EVENTLOG_API EventLogEntry : public MatchExpression::Matchable
{
    public:
        int contextModuleId;
        int level; // indent level (method call depth)

    protected:
        Event* event; // back pointer
        int entryIndex;
        static char buffer[128];
        static LineTokenizer tokenizer; // not thread safe

    public:
        EventLogEntry();
        virtual ~EventLogEntry() {}
        virtual void parse(char *line, int length) = 0;
        virtual void print(FILE *fout) = 0;
        virtual int getClassIndex() = 0;
        virtual const char *getClassName() = 0;

        Event *getEvent() { return event; }
        int getEntryIndex() { return entryIndex; }

        virtual const std::vector<const char *> getAttributeNames() const = 0;
        virtual const char *getAsString() const = 0;
        virtual const char *getAsString(const char *attribute) const = 0;

        static EventLogEntry *parseEntry(EventLog *eventLog, Event *event, int entryIndex, file_offset_t offset, char *line, int length);
        static eventnumber_t parseEventNumber(const char *str);
        static simtime_t parseSimulationTime(const char *str);
};

/**
 * Base class for entries represented by key value tokens.
 */
class EVENTLOG_API EventLogTokenBasedEntry : public EventLogEntry
{
    public:
        static char *getToken(char **tokens, int numTokens, const char *sign, bool mandatory);
        static bool getBoolToken(char **tokens, int numTokens, const char *sign, bool mandatory, bool defaultValue);
        static int getIntToken(char **tokens, int numTokens, const char *sign, bool mandatory, int defaultValue);
        static short getShortToken(char **tokens, int numTokens, const char *sign, bool mandatory, short defaultValue);
        static long getLongToken(char **tokens, int numTokens, const char *sign, bool mandatory, long defaultValue);
        static int64 getInt64Token(char **tokens, int numTokens, const char *sign, bool mandatory, int64 defaultValue);
        static eventnumber_t getEventNumberToken(char **tokens, int numTokens, const char *sign, bool mandatory, eventnumber_t defaultValue);
        static simtime_t getSimtimeToken(char **tokens, int numTokens, const char *sign, bool mandatory, simtime_t defaultValue);
        static const char *getStringToken(char **tokens, int numTokens, const char *sign, bool mandatory, const char *defaultValue);

    public:
        virtual void parse(char *line, int length);
        virtual void parse(char **tokens, int numTokens) = 0;
};

/**
 * One line log message entry.
 */
class EVENTLOG_API EventLogMessageEntry : public EventLogEntry
{
    public:
        const char *text;

    public:
        EventLogMessageEntry(Event *event, int entryIndex);
        virtual void parse(char *line, int length);
        virtual void print(FILE *fout);
        virtual int getClassIndex() { return 0; }
        virtual const char *getClassName() { return "EventLogMessageEntry"; }

        virtual const std::vector<const char *> getAttributeNames() const;
        virtual const char *getAsString() const { return "-"; }
        virtual const char *getAsString(const char *attribute) const;
};

NAMESPACE_END


#endif
