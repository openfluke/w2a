package go_v_metal_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Publication is tested with fake git/gh commands: no real refs or remotes change.
func TestMetalReleaseScript(t *testing.T) {
	root, e := filepath.Abs("../../../welvet.metal")
	if e != nil {
		t.Fatal(e)
	}
	for _, dry := range []bool{true, false} {
		name := "publish_mock"
		if dry {
			name = "dry_run_no_mutation"
		}
		t.Run(name, func(t *testing.T) {
			fixture := t.TempDir()
			bin := filepath.Join(fixture, "mockbin")
			if e = os.MkdirAll(bin, 0755); e != nil {
				t.Fatal(e)
			}
			for _, name := range []string{"release.sh", "CMakeLists.txt", "README.md", "LICENSE", "VERSIONING.md", "src/interop/c_abi.cpp", "docs/GO_INTEGRATION.md", "docs/METAL_ONLY.md"} {
				data, e := os.ReadFile(filepath.Join(root, name))
				if e != nil {
					t.Fatal(e)
				}
				path := filepath.Join(fixture, name)
				if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(path, data, 0644); e != nil {
					t.Fatal(e)
				}
			}
			for _, name := range []string{"shared/libwelvet_metal.dylib", "shared/welvet_dense_fwd", "static/libwelvet_metal.a"} {
				path := filepath.Join(fixture, "build-release", name)
				if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(path, []byte("fixture"), 0644); e != nil {
					t.Fatal(e)
				}
			}
			mock := `#!/usr/bin/env python3
import json,os,pathlib,sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
root=pathlib.Path(os.environ['RELEASE_FIXTURE'])
with (root/'calls.jsonl').open('a') as f: f.write(json.dumps([name,args])+'\n')
if name=='git':
 if args[:1]==['symbolic-ref']:print('main')
 elif args[:2]==['remote','get-url']:print('https://github.com/openfluke/welvet.metal.git')
 elif args[:1]==['status']:
  if not (root/'committed').exists():print(' M release.sh')
 elif args[:1]==['rev-parse']:
  if '--verify' in args:sys.exit(0 if (root/'tagged').exists() else 1)
  print('f'*40)
 elif args[:1]==['commit']:(root/'committed').touch()
 elif args[:1]==['tag']:(root/'tagged').touch()
 elif args[:1] not in [['add'],['push']]:raise SystemExit('unexpected git command '+str(args))
elif name=='gh':pass
elif name=='cmake':
 if args[0]!='--install':raise SystemExit('unexpected CMake build')
 p=pathlib.Path(args[args.index('--prefix')+1]);(p/'lib').mkdir(parents=True,exist_ok=True)
 if args[1].endswith('shared'):
  (p/'lib/libwelvet_metal.dylib').write_text('fixture')
  (p/'bin').mkdir(exist_ok=True);tool=p/'bin/welvet_dense_fwd';tool.write_text('#!/bin/sh\necho 0.68\n');tool.chmod(0o755)
 else:(p/'lib/libwelvet_metal.a').write_text('fixture')
elif name=='otool':print('fixture:\n @rpath/libwelvet_metal.dylib (compatibility version 0.0.0)')
elif name=='xcrun':print('27.0')
elif name=='clang++':print('Apple clang release fixture')
elif name=='uname':print('Darwin' if args==['-s'] else 'arm64')
elif name=='sysctl':print('4')
else:raise SystemExit('unexpected tool '+name)
`
			for _, name := range []string{"git", "gh", "cmake", "otool", "xcrun", "clang++", "uname", "sysctl"} {
				if e = os.WriteFile(filepath.Join(bin, name), []byte(mock), 0755); e != nil {
					t.Fatal(e)
				}
			}
			args := []string{filepath.Join(fixture, "release.sh"), "--skip-build", "--skip-tests"}
			if dry {
				args = append(args, "--dry-run")
			}
			cmd := exec.Command("/bin/bash", args...)
			cmd.Env = append(os.Environ(), "PATH="+bin+":"+os.Getenv("PATH"), "RELEASE_FIXTURE="+fixture)
			if out, e := cmd.CombinedOutput(); e != nil {
				t.Fatalf("%v\n%s", e, out)
			}
			calls, e := os.ReadFile(filepath.Join(fixture, "calls.jsonl"))
			if e != nil {
				t.Fatal(e)
			}
			mutations, release := 0, false
			for _, line := range strings.Split(strings.TrimSpace(string(calls)), "\n") {
				var item []json.RawMessage
				if e = json.Unmarshal([]byte(line), &item); e != nil {
					t.Fatal(e)
				}
				var name string
				var args []string
				json.Unmarshal(item[0], &name)
				json.Unmarshal(item[1], &args)
				if name == "git" && len(args) > 0 && (args[0] == "add" || args[0] == "commit" || args[0] == "tag" || args[0] == "push") {
					mutations++
				}
				if dry && name == "gh" {
					t.Fatal("dry run invoked GitHub CLI")
				}
				if name == "gh" && len(args) > 1 && args[0] == "release" && args[1] == "create" {
					release = true
					joined := strings.Join(args, " ")
					if !strings.Contains(joined, "--verify-tag") || !strings.Contains(joined, "--notes-file") || !strings.Contains(joined, "v0.68") {
						t.Fatal("incomplete release command", args)
					}
				}
			}
			if dry && (mutations != 0 || release) {
				t.Fatal("dry run changed git/publication state")
			}
			if !dry && (mutations != 5 || !release) {
				t.Fatalf("mock publication incomplete: mutations=%d release=%v", mutations, release)
			}
			if _, e = os.Stat(filepath.Join(fixture, "dist/welvet-metal-v0.68-macos-arm64.tar.gz")); e != nil {
				t.Fatal(e)
			}
		})
	}
}
