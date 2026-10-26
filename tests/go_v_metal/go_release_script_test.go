package go_v_metal_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestGoReleaseScriptMarkdown(t *testing.T) {
	root, e := filepath.Abs("../../../welvet")
	if e != nil {
		t.Fatal(e)
	}
	readme, e := os.ReadFile(filepath.Join(root, "README.md"))
	if e != nil {
		t.Fatal(e)
	}
	version := regexp.MustCompile(`\*\*Version\*\*\s*\|\s*\*\*(v[\d.]+)\*\*`).FindSubmatch(readme)
	if len(version) != 2 {
		t.Fatal("version missing")
	}
	tag := string(version[1])
	for _, mode := range []string{"dry", "notes", "create_existing_tag", "edit_existing_release", "create_new_tag"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			for _, name := range []string{"release.sh", "README.md"} {
				data, e := os.ReadFile(filepath.Join(root, name))
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(filepath.Join(dir, name), data, 0644); e != nil {
					t.Fatal(e)
				}
			}
			bin := filepath.Join(dir, "mockbin")
			os.MkdirAll(bin, 0755)
			mock := `#!/usr/bin/env python3
import json,os,pathlib,sys
root=pathlib.Path(os.environ['RELEASE_FIXTURE']);name=pathlib.Path(sys.argv[0]).name;args=sys.argv[1:]
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps([name,args])+'\n')
if name=='git':
 if args[:1]==['status']:pass
 elif args[:1]==['rev-parse']:sys.exit(1 if os.environ['RELEASE_MODE']=='create_new_tag' else 0)
 elif args[:1] not in [['push'],['tag']]:raise SystemExit('unexpected git mutation '+str(args))
elif name=='gh':
 if args[:2]==['release','view']:sys.exit(0 if os.environ['RELEASE_MODE']=='edit_existing_release' else 1)
 if args[:2] not in [['release','create'],['release','edit']]:raise SystemExit('unexpected gh command')
 p=pathlib.Path(args[args.index('--notes-file')+1]);(root/'captured-notes.md').write_text(p.read_text())
`
			for _, name := range []string{"git", "gh"} {
				os.WriteFile(filepath.Join(bin, name), []byte(mock), 0755)
			}
			args := []string{filepath.Join(dir, "release.sh"), "--skip-build"}
			if mode == "dry" {
				args = append(args, "--dry-run")
			}
			if mode == "notes" {
				args = append(args, "--notes-only")
			}
			cmd := exec.Command("/bin/bash", args...)
			cmd.Env = append(os.Environ(), "PATH="+bin+":"+os.Getenv("PATH"), "RELEASE_FIXTURE="+dir, "RELEASE_MODE="+mode)
			if out, e := cmd.CombinedOutput(); e != nil {
				t.Fatalf("%v\n%s", e, out)
			}
			notes, e := os.ReadFile(filepath.Join(dir, "dist", "release-"+tag+".md"))
			if e != nil {
				t.Fatal(e)
			}
			text := string(notes)
			for _, literal := range []string{"## Welvet " + tag, "\n```bash\n", "`nativemetal`", "`SyncToGo()`", "`EnableRouting`", "go get github.com/openfluke/welvet@" + tag} {
				if !strings.Contains(text, literal) {
					t.Fatalf("missing literal %q: %s", literal, text)
				}
			}
			if strings.Contains(text, "\\`") || strings.Contains(text, "${tag}") || strings.Contains(text, "@VERSION@") {
				t.Fatal("unrendered Markdown or version marker", text)
			}
			calls, e := os.ReadFile(filepath.Join(dir, "calls.jsonl"))
			if mode == "dry" || mode == "notes" {
				if e == nil && len(calls) > 0 {
					t.Fatal("preview invoked git/GitHub", string(calls))
				}
				return
			}
			if e != nil {
				t.Fatal(e)
			}
			captured, e := os.ReadFile(filepath.Join(dir, "captured-notes.md"))
			if e != nil {
				t.Fatal(e)
			}
			if string(captured) != text {
				t.Fatal("GitHub received different notes")
			}
			for _, line := range strings.Split(strings.TrimSpace(string(calls)), "\n") {
				var item []json.RawMessage
				json.Unmarshal([]byte(line), &item)
				var args []string
				json.Unmarshal(item[1], &args)
				for _, arg := range args {
					if arg == "--force" {
						t.Fatal("unexpected force push")
					}
				}
				if mode != "create_new_tag" && len(args) > 0 && args[0] == "tag" {
					t.Fatal("existing version tag was recreated")
				}
			}
		})
	}
}
