import React, { memo } from 'react';
import Recorder from './Recorder';
import FileConverter from './FileConverter';

/** Library: the takes recorded in Live, and converting whole files. */
function Library({ pipeline, converter }) {
  const count = pipeline.library.takes.length;
  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-2">
      <section className="panel flex min-h-80 flex-col p-4 lg:min-h-0" aria-labelledby="recordings-title">
        <h2 id="recordings-title" className="flex h-7 shrink-0 items-center gap-2 text-sm font-semibold text-fg">
          Recordings
          {count ? <span className="font-mono text-xs font-normal text-fg-subtle">{count}</span> : null}
        </h2>
        <div className="mt-1 flex min-h-0 flex-1 flex-col">
          <Recorder pipeline={pipeline} />
        </div>
      </section>
      <FileConverter {...converter} />
    </div>
  );
}

export default memo(Library);
