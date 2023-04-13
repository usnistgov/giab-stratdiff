# Stratcomp

Compare stritifications. This is meant as a diagnostic tool to assess changes
b/t different stratifications versions.

## Usage

Simple comparison:

```
./stratcomp path/to/stratifications/a path/to/stratifications/b out
```

### Mapping file

Since stratifications may be named differently, use a mapping file to tell the
tool which file names represent "the same" bed file. This is a tsv file like so:

```
nameA nameB
...   ...
```

Where names in columns 1 and 2 correspond to the first and second
stratifications given to the tool respectively.

