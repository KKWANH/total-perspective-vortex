# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kkim <kkim@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/31 16:01:01 by kkim              #+#    #+#              #
#    Updated: 2023/02/10 12:21:56 by kkim             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Color
BLA	= \033[30m
RED	= \033[31m
GRE	= \033[32m
YEL	= \033[33m
BLU	= \033[34m
MAG	= \033[35m
CYA	= \033[36m
WHI	= \033[37m
DEF	= \033[38m

# Font Style
BLD	= \033[1m
UND	= \033[4m

# Reset
RES	= \033[0m

# Defines

# Main
all:
	@make help

setup:
	@printf "$(BLD)$(GRE)[Setup]   $(RES)\tSetting virtual-environment by $(CYA)[setup.sh]$(RES).\n"
	@printf "$(BLD)$(GRE)          $(RES)\t$(MAG)[pandas]$(RES).\n"
	@printf "$(BLD)$(GRE)          $(RES)\t$(MAG)[numpy]$(RES).\n"
	@printf "$(BLD)$(GRE)          $(RES)\t$(MAG)[matplotlib]$(RES).\n"
	@printf "$(BLD)$(GRE)          $(RES)\t$(MAG)[mne]$(RES).\n"
	@printf "$(BLD)$(GRE)          $(RES)\t$(MAG)[sklearn]$(RES).\n"
	@sh "srcs/setup.sh" > /dev/null
	@printf "$(BLD)$(GRE)          $(RES)\tPlease                                              $(MAG)[sklearn]$(RES).\n"


help:
	@printf "$(BLD)$(YEL)[Help]    $(RES)\tWelcome to $(BLD)$(UND)$(CYA)Total-Perpective-Vortex$(RES) by $(BLD)$(MAG)kkim, seongcho$(RES).\n"
	@printf "$(BLD)$(YEL)          $(RES)\tthere are $(YEL)[]$(RES) options.\n"