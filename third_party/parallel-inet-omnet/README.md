OMNeT++ for Parallel INET
~~~~~~~~~~~~~~~~~~~~~~~~~

This directory contains a modified version of OMNeT++. Modifications primarily target
features used by our modified version of INET. The purpose of these modifications is
to allow executing INET models in parallel.

We grant usage of our modifications under the same licenses as OMNeT++ 4.4 is licensed.

Mirko Stoffers, Ralf Bettermann, James Gross, Klaus Wehrle
RWTH Aachen University, Chair of Communication and Distributed Systems

OMNeT++
~~~~~~~

OMNeT++ is a public-source, component-based, modular and open-architecture
simulation environment with strong GUI support and an embeddable simulation
kernel. Its primary application area is the simulation of communication
networks, but it has been successfully used in other areas like the simulation
of IT systems, queueing networks, hardware architectures and business processes
as well.

OLD USERS:
	If you have simulations written for OMNeT++ 3.x that you want to
	port to 4.x, see the MIGRATION file, doc/migration.pdf, and the
	scripts under migrate/!

If you installed the source distribution, the omnetpp directory on your system
should contain the following subdirectories. (If you installed a precompiled
distribution, some of the directories may be missing, or there might be
additional directories, e.g. containing software bundled with OMNeT++.)

The simulation system itself:

  omnetpp/         OMNeT++ root directory
    bin/           OMNeT++ executables (opp_run, nedtool, scavetool, etc.)
    include/       header files for simulation models
    lib/           library files
    images/        icons that can be used in network graphics
    doc/           manual (PDF), readme, license, etc.
      manual/      manual in HTML
      tictoc-tutorial/  introduction into using OMNeT++
      api/         API reference in HTML
      nedxml-api/  API reference for the NEDXML library
      src/         sources of the documentation
    src/           OMNeT++ sources
      sim/         simulation kernel
        parsim/    files for distributed execution
        netbuilder/files for dynamically reading NED files
      envir/       common code for runtime user interfaces
      cmdenv/      command-line runtime user interface
      tkenv/       Tcl/Tk-based graphical runtime user interface
      nedxml/      nedtool, message compiler, NED infrastructure
      layout/      graph layouting library
      scave/       library for processing result files
      eventlog/    library for processing event log files
      common/      common utility classes
      utils/       makefile generator and various utilities
    ide/           the OMNeT++ integrated environment
    migrate/       scripts for migrating simulation models from the 3.x version
    test/          regression test suite
      core/        regression test suite for the simulation library
      distrib/     regression test suite for built-in distributions
      ...

Sample simulations are in the samples directory.

    samples/     directories for sample simulations
      aloha/     models the Aloha protocol
      cqn/       Closed Queueing Network
      ...

The contrib directory contains material from the OMNeT++ community.

    contrib/     directory for contributed material
      jsimplemodule/  package for writing OMNeT++ simulations in Java
      jresultwriter/  package for creating OMNeT++ result files from 3rd party Java-based simulators
      topologyexport/ simple module for exporting model topology in XML
      akaroa/    patch file to compile akaroa with GCC 4.4
      gtk/       files to make Eclipse look good on Linux
      octave/    Octave scripts for result processing (somewhat outdated)
      emacs/     NED syntax highlighting for Emacs (somewhat outdated)
      med/       NED syntax highlighting for the MED editor (somewhat outdated)


The example simulations
~~~~~~~~~~~~~~~~~~~~~~~

The example simulations are designed to demonstrate many of the features of
OMNeT++. We recommend that you try tictoc first, which also has an
accompanying tutorial under doc/.


Installation
~~~~~~~~~~~~

Please see the Install Guide, doc/InstallGuide.pdf for specific instructions
for various operating systems, and to read about dependencies, optional 
packages and build options.

Enjoy.

Andras Varga and the OMNeT++ Team.
