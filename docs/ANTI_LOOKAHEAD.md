# Anti-look-ahead rule

For target date `D`, live forecast training is restricted to observed draws from the 730-calendar-day interval ending at `D - 1`. If `D` is already present in the upstream result source, the live forecast command aborts instead of creating a retrospective snapshot.
