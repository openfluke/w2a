package go_v_cuda

import "strings"

func withTypedVolumes(base, parent []Case, prefix string) []Case {
	for _, c := range parent {
		if strings.Contains(c.Name, "volume 211 × all 34 dtypes") {
			base = append(base, Case{Name: prefix + "volume alias / " + c.Name, Run: c.Run})
		}
	}
	return base
}
