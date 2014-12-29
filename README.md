%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                The style file of International conference                 %%
%%             Intelligent Information Processing, IIP-9 (2012)              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%               copyleft (GPL)      K.V.Vorontsov, 2007-2012                %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{jmlda}[2012/21/12 Journal of Machine Learning and Data Analysis]

% Опция draft: выводить замечания, доп.информацию в оглавлении, статистику
\newif\ifdraft\draftfalse
\DeclareOption{draft}{\drafttrue}

% Опция writebat: выводить bat-файлы для формирования архивов
\newif\ifwritebat\writebatfalse
\DeclareOption{writebat}{\writebattrue}

% Опция writetab: выводить tab-файл со списком статей
\newif\ifwritetab\writetabfalse
\DeclareOption{writetab}{\writetabtrue}

% Опция writehtml: выводить html-файл со списком статей
\newif\ifwritehtml\writehtmlfalse
\DeclareOption{writehtml}{\writehtmltrue}

\ProcessOptions

\RequirePackage[cp1251]{inputenc}
%\RequirePackage[T2A]{fontenc}
\RequirePackage{amssymb}
\RequirePackage{amsmath}
\RequirePackage{mathrsfs}
\RequirePackage{euscript}
\RequirePackage{upgreek}
\RequirePackage[english,russian]{babel}
\RequirePackage[displaymath,mathlines]{lineno} 
\RequirePackage{array}
\RequirePackage{theorem}
\RequirePackage[ruled]{algorithm}
\RequirePackage[noend]{algorithmic}
\RequirePackage[all]{xy}
\RequirePackage{graphicx}
\RequirePackage{epstopdf}   % for Mikhail Burmistrov burmisha@gmail.com
\RequirePackage{tikz}       % for Mikhail Burmistrov burmisha@gmail.com
\RequirePackage{pgfplots}   % for Mikhail Burmistrov burmisha@gmail.com
%\RequirePackage{psfrag}  % Pytiev_3
%\RequirePackage{tikz}\usetikzlibrary{positioning}   % yangel
%\RequirePackage{epic}
\RequirePackage{color}
%\RequirePackage[footnotesize]{caption2}
\RequirePackage{ifthen}
\RequirePackage{url}
%\RequirePackage{html}
%\RequirePackage[colorlinks,urlcolor=blue]{hyperref}
\RequirePackage{makeidx}
\RequirePackage{pb-diagram}
\RequirePackage{subfig}
%\RequirePackage{lamsarrow, pb-lams, stmaryrd}  %% Gurov
%% Двухколоночный набор:
\RequirePackage{balance}  %% неверно обрабатываются сноски на балансируемой странице
%\newcommand{\balance}{\relax}
%\newcommand{\nobalance}{\relax}
%\RequirePackage{multicol} %% глотает плавающие иллюстрации
%\RequirePackage{dblfloatfix}  %% никакого эффекта!
%\RequirePackage{cuted}  %% генерируется ошибка
\RequirePackage{multirow} %% для Генрихова

\renewcommand{\baselinestretch}{1}
%\renewcommand{\baselinestretch}{1.1} %для печати с большим интервалом

\renewcommand{\thesubfigure}{\asbuk{subfigure}}

% Для A5
%\textheight=175mm
%\headsep=5mm
%\textwidth=117mm
%\topmargin=25mm
%\oddsidemargin=20mm
%\evensidemargin=20mm

% Для А4 в две колонки
\textheight=240mm
\textwidth=170mm
\oddsidemargin=0mm
\evensidemargin=-11mm
\topmargin=-10mm
\footnotesep=3ex
\columnsep=6mm

\marginparwidth=5mm
\parindent=3.5ex
\tolerance=9000
\hbadness=2000
\flushbottom
%\raggedbottom
% подавить эффект "висячих стpок"
\clubpenalty=10000
\widowpenalty=10000

%% Печать во весь лист
%% INTERFACE
%\def\MinMargins{
%    \textheight=240mm
%    \textwidth=160mm
%    \oddsidemargin=5mm
%    \evensidemargin=5mm
%    \topmargin=-10mm
%    \footnotesep=3ex
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Языковые настройки
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\ruseng}[2]{\iflanguage{russian}{#1}\relax\iflanguage{english}{#2}\relax}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление заголовков статей
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newlength\vskipBeforeTitle
\newlength\vskipAfterTitle

% сделать неразрывные пробелы ещё и нерастяжимыми (например между фамилией и инициалами авторов)
\newcommand\unstretchspaces{\catcode`~=\active\def~{\;}}

% Настройки заголовка статьи
% INTERFACE
\setlength\vskipBeforeTitle{3ex}
\setlength\vskipAfterTitle{3ex}
\newcommand\typeTitle[1]{{\Large\sffamily\bfseries #1}}
\newcommand\typeAuthor[1]{{\itshape\bfseries #1}}
\newcommand\typeOrganization[1]{{\small #1}}
\newcommand\typeEmail[1]{{\ttfamily #1}}
\newcommand\typeTocAuthorTitle[2]{{\unstretchspaces\itshape #1}\\ #2}
\newcommand\typeAbstract[1]{%
    \par\vskip\vskipAfterTitle
    \noindent\parbox{159.3mm}{\small #1}%
}

% Вывод заголовка
\def\type@title{%
    \noindent
    \begingroup
    % подготовка к выводу сноски с грантом
    \setcounter{footnote}{0}%
    \def\thefootnote{\fnsymbol{footnote}}%
    %\def\@makefnmark{\hbox to 0pt{$^{\@thefnmark}$\hss}}%
    \def\@makefnmark{$^{\@thefnmark}$}%
    %\long\def\@makefntext##1{\parindent 1em\noindent\hbox to1.8em{\hss $\m@th ^{\@thefnmark}$}##1}%
    \def\@makefntext##1{\noindent $\m@th ^{\@thefnmark}$##1}%
    % вывод одноколоночного заголовка статьи, включая аннотацию
%    \twocolumn[%
        \parbox{\textwidth}{%
            \begin{center}%
                \vskip-6ex\vskip\vskipBeforeTitle%
                %--------- основной заголовок
                \ifthenelse{\equal{\@title}{}}{}%
                    {\typeTitle\@title\ifthenelse{\equal{\@thanks@grant}{}}{}{\footnotemark\setcounter{footnote}{0}}\\[1ex]}%
                \ifthenelse{\equal{\@author@long}{}}{}%
                    {\typeAuthor\@author@long}%
                \ifthenelse{\equal{\@email}{}}{}%
                    {\\{\typeEmail\@email}}%
                \ifthenelse{\equal{\@organization}{}}{}%
                    {\\{\typeOrganization\@organization}}%
                \ifthenelse{\equal{\@abstract}{}}{}%
                    {\typeAbstract\@abstract\par\vskip\vskipAfterTitle}%
                \ruseng{{%
                %--------- второй заголовок на английском для статей на русском
                \English%
                \ifthenelse{\equal{\@title@eng}{}}{}%
                    {\typeTitle\@title@eng\ifthenelse{\equal{\@thanks@grant}{}}{}{\footnotemark\setcounter{footnote}{0}}\\[1ex]}%
                \ifthenelse{\equal{\@author@long@eng}{}}{}%
                    {\typeAuthor\@author@long@eng}%
                \ifthenelse{\equal{\@organization@eng}{}}{}%
                    {\\{\typeOrganization\@organization@eng}}%
                \ifthenelse{\equal{\@abstract@eng}{}}{}%
                    {\typeAbstract\@abstract@eng}%
                }}{{%
                %--------- второй заголовок на русском для статей на английском
                \Russian%
                \ifthenelse{\equal{\@title@rus}{}}{}%
                    {\typeTitle\@title@rus\ifthenelse{\equal{\@thanks@grant}{}}{}{\footnotemark}\\[1ex]}%
                \ifthenelse{\equal{\@author@long@rus}{}}{}%
                    {\typeAuthor\@author@long@rus}%
                \ifthenelse{\equal{\@organization@rus}{}}{}%
                    {\\{\typeOrganization\@organization@rus}}%
                \ifthenelse{\equal{\@abstract@rus}{}}{}%
                    {\typeAbstract\@abstract@rus}%
                }}%
            \end{center}%
            \vskip-2ex\vskip\vskipAfterTitle
        }%
%    ]%
    % помещение сноски вниз страницы
    \ifthenelse{\equal{\@thanks@grant}{}}{}{\footnotetext{\@thanks@grant}}%
    \setcounter{footnote}{0}%
    \endgroup
}
\def\maketitle{%
    \@BeginDocument
    % на предыдущей странице почему-то подавлялся вывод колонтитулов
    % (видимо, это побочный эффект \twocolumn
    \pagestyle{headings}%
    %\clearpage
    % печать заголовка вместе с аннотацией
    \type@title
    \globallabel{\@paper@name:begin}%
    \pagestyle{headings}%
    \thispagestyle{headfoot}%
    % обработать список фамилий авторов, занести каждую в авторский указатель
    %\newcommand\@trim[1]{\@ifnextchar\ \@gobble\relax#1}%
    \@for\@indx@elem:=\@author\do{%
        \index{\@indx@elem}%
    }%
    % в чистовом режиме записать в оглавление только авторов и название
    \addcontentsline{toc}{mmrotitle}{%
        \typeTocAuthorTitle{\@author}{\@title}%
    }%
    % в черновом режиме дописать в оглавление:
    % организацию, абстракт, имя файла, имена рецензентов, отметки о прохождении корректуры и рецензирования
    \ifdraft{\Russian
        \addtocontents{toc}{\protect\par(\@organization)}%
        \addtocontents{toc}{\protect\par\@abstract}%
        \ifthenelse{\equal{\@paper@corrected}{+}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\bigstar$\quad
                \textsl{Статья прошла корректуру}}}%
            {}%
        \ifthenelse{\equal{\@paper@reviewed}{+}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\bigstar\bigstar$\quad
                \textsl{Статья прошла рецензирование}}}%
            {}%
        \ifthenelse{\equal{\@paper@accepted}{0}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak ???\quad
                \textsl{Решение по данной статье пока не принято}}}%
            {}%
        \ifthenelse{\equal{\@paper@accepted}{+}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\bigstar\bigstar\bigstar$\quad
                \textsl{Статья принята в печать}}}%
            {}%
        \ifthenelse{\equal{\@paper@accepted}{?}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\maltese$\quad
                \textbf{Статью желательно доработать}}}%
            {}%
        \ifthenelse{\equal{\@paper@accepted}{!}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\maltese\maltese$\quad
                \textbf{Статья не может быть принята без доработки}}}%
            {}%
        \ifthenelse{\equal{\@paper@accepted}{-}}%
            {\addtocontents{toc}{\protect\par\protect\nopagebreak$\maltese\maltese\maltese$\quad
                \textbf{Статья отвергнута}}}%
            {}%
        \addtocontents{toc}{%
            \protect\par\protect\url{\@paper@name.TeX}%
            \protect\ifthenelse{\protect\equal{\@reviewers@list}{}}%
                {}%
                {\quad(рецензенты и корректоры: \texttt{\@reviewers@list})%
            }%
        }%
    }\fi
    % сформировать колонтитулы
    \markboth{\@author@short}{\@title@short}%
    \par\nobreak\@afterheading
    % балансировка колонок -- только в конце статьи
    \nobalance
}
\def\@clear@title{%
    \gdef\@title{}%
    \gdef\@title@short{}%
    \gdef\@author{}%
    \gdef\@author@short{}%
    \gdef\@author@long{}%
    \gdef\@organization{}%
    \gdef\@abstract{}%
    \gdef\@thanks@grant{}%
    \gdef\@email{}%
    %--------- второй заголовок на английском для статей на русском
    \gdef\@title@eng{}%
    \gdef\@title@short@eng{}%
    \gdef\@author@eng{}%
    \gdef\@author@short@eng{}%
    \gdef\@author@long@eng{}%
    \gdef\@organization@eng{}%
    \gdef\@abstract@eng{}%
    %--------- второй заголовок на русском для статей на английском
    \gdef\@title@rus{}%
    \gdef\@title@short@rus{}%
    \gdef\@author@rus{}%
    \gdef\@author@short@rus{}%
    \gdef\@author@long@rus{}%
    \gdef\@organization@rus{}%
    \gdef\@abstract@rus{}%
}
\renewcommand{\title}[2][]{\@clear@title
    \gdef\@title{#2}%    {\uppercase{#2}}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@title@short{{#2}}}%
        {\gdef\@title@short{{#1}}}%
}
\renewcommand{\author}[2][]{
    \gdef\@author{#2}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@author@short{{#2}}}%
        {\gdef\@author@short{{#1 и~др.}}}%
    \gdef\@author@long{#2}%
    \@ifnextchar[\set@author@long\relax
}
\def\set@author@long[#1]{\gdef\@author@long{#1}}
\newcommand{\organization}[1]{\gdef\@organization{{#1}}}
\newcommand{\email}[1]{\gdef\@email{{#1}}}
\renewcommand{\thanks}[1]{\gdef\@thanks@grant{{#1}}}
%\def\myuncat{\def\do##1{\catcode`##1=12}\dospecials}
\renewcommand{\abstract}[1]{\gdef\@abstract{{#1}}}

% То же самое для второго заголовка на английском
\newcommand{\titleEng}[2][]{%
    \gdef\@title@eng{#2}%    {\uppercase{#2}}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@title@short@eng{{#2}}}%
        {\gdef\@title@short@eng{{#1}}}%
}
\newcommand{\authorEng}[2][]{
    \gdef\@author@eng{#2}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@author@short@eng{{#2}}}%
        {\gdef\@author@short@eng{{#1 и~др.}}}%
    \gdef\@author@long@eng{#2}%
    \@ifnextchar[\set@author@long@eng\relax
}
\def\set@author@long@eng[#1]{\gdef\@author@long@eng{#1}}
\newcommand{\organizationEng}[1]{\gdef\@organization@eng{{#1}}}
\newcommand{\abstractEng}[1]{\gdef\@abstract@eng{{#1}}}

% То же самое для второго заголовка на русском
\newcommand{\titleRus}[2][]{%
    \gdef\@title@rus{#2}%    {\uppercase{#2}}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@title@short@rus{{#2}}}%
        {\gdef\@title@short@rus{{#1}}}%
}
\newcommand{\authorRus}[2][]{
    \gdef\@author@rus{#2}
    \ifthenelse{\equal{#1}{}}%
        {\gdef\@author@short@rus{{#2}}}%
        {\gdef\@author@short@rus{{#1 и~др.}}}%
    \gdef\@author@long@rus{#2}%
    \@ifnextchar[\set@author@long@rus\relax
}
\def\set@author@long@rus[#1]{\gdef\@author@long@rus{#1}}
\newcommand{\organizationRus}[1]{\gdef\@organization@rus{{#1}}}
\newcommand{\abstractRus}[1]{\gdef\@abstract@rus{{#1}}}

% Библиографическая запись текущей статьи
\def\paper@record{%{%
%    %\textit{\@author} \@title~//
%    \Russian
%    Машинное обучение и анализ данных, 2013. Т.\,1, \No\,5.\\
%    \English
%    Machine Learning and Data Analysis, 2013. Vol.\,1\,(5).}%
%    %М.:~МАКС Пресс, 2009~---
%    %С.\,\globalpageref{\@paper@name:begin}--\globalpageref{\@paper@name:end}.
}

% Оформление аннотации
%\renewenvironment{abstract}{\begin{strip}}{\end{strip}}
%\renewenvironment{abstract}{\par\begingroup\small}{\endgroup\par\medskip}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Переопределение колонтитулов
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\type@chapter@code{%
    \ifthenelse{\equal{\@chapter@code}{}}{}{\typeChapterCode{\@chapter@code}}
}
\renewcommand{\ps@headings}{%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\footnotesize
        \rightmark\hfill\type@chapter@code\quad\thepage\\[-2ex]\hrule}}%
    \renewcommand{\@evenhead}{\parbox{\textwidth}{\footnotesize
        \thepage\quad\type@chapter@code\hfill\leftmark\\[-2ex]\hrule}}%
    \renewcommand{\@oddfoot}{}%
    \renewcommand{\@evenfoot}{}%
}
\newcommand{\ps@headfoot}{%
    \renewcommand{\@oddhead}{\parbox{\textwidth}{\footnotesize
        \rightmark\hfill\type@chapter@code\quad\thepage\\[-2ex]\hrule}}%
    \renewcommand{\@evenhead}{\parbox{\textwidth}{\footnotesize
        \thepage\quad\type@chapter@code\hfill\leftmark\\[-2ex]\hrule}}%
    \renewcommand{\@oddfoot}{\parbox{\textwidth}{\scriptsize\slshape\paper@record}}%\hrule\vskip1ex
    \renewcommand{\@evenfoot}{\@oddfoot}%
}
\pagestyle{empty}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление разделов
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Настройка разделов
% INTERFACE
\newcommand\typeSection[1]{%
    \medskip
    \hangindent=3.5ex
    \hangafter=-6\noindent
    {\large\sffamily\bfseries #1}%
    \par\nobreak\smallskip
}
% INTERFACE
\newcommand\typeParagraph[1]{%
    \smallskip{\normalfont\rmfamily\bfseries #1 }%
}
% INTERFACE
\newcommand\typeChapter[3]{%
    \hrule\vskip1pt\hrule height2pt\vskip4ex\noindent
    {\normalfont\LARGE\sffamily\bfseries #3\raggedright\par}
    \vskip2ex\hrule height2pt\vskip1pt\hrule\vskip-1ex
    {\footnotesize\flushright\Russian Код раздела: #1 (#2)\par}
    \vskip6ex
}
% INTERFACE
\newcommand\typeChapterCode[1]{({\tiny\Russian JMLDA})}

% Разделы
\renewcommand\section[1]{\par\typeSection{#1}\@afterheading}
\renewcommand\subsection[1]{\par\typeSection{#1}\@afterheading}
\renewcommand\subsubsection[1]{\par\typeSection{#1}\@afterheading}
\renewcommand\paragraph[1]{\par\typeParagraph{#1}\nobreak}
\renewcommand\subparagraph[1]{\par\typeParagraph{#1}\nobreak}

% Начало новой главы
\def\newchapterpage{%
    \pagestyle{headings}
    \onecolumn
%%%    \ifthenelse{\isodd{\thepage}}%s
%%%        {\newpage~\thispagestyle{empty}\newpage}    % \renewcommand{\@evenhead}{}
%%%        {\newpage}%
%    \ifthenelse{\isodd{\thepage}}%s
%        {~\thispagestyle{empty}\newpage}    % \renewcommand{\@evenhead}{}
%        {}%
    \thispagestyle{empty}%
}
\def\chaptercode#1{\gdef\@chapter@code{#1}}
\chaptercode{??}
% INTERFACE
\def\chapter#1#2#3{%
    \newchapterpage
    %\typeChapter{#1}{#2}{#3}
    \addcontentsline{toc}{mmrochapter}{{\normalfont\sffamily\bfseries #3}}% \protect\chaptercode{#1}
    \addtocontents{toc}{\protect\nopagebreak}%
    \addcontentsline{tos}{mmrochapter}{\rlap{(#1)}\hspace{12mm}#2\protect\\\strut\hspace{7.3mm}#3}% для краткого оглавления
    \markboth{#3}{#3}%
    \chaptercode{#1}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление оглавления
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\contentsname{Содержание}

% вёрстка одного пункта оглавления
\newcommand\l@mmrotitle{\smallskip\@dottedtocline{1}{0ex}{3ex}}
\newcommand\l@mmrochapter{\vskip3ex\@dottedtocline{0}{0ex}{3ex}}

% переопределение команды генерации оглавления
\renewcommand\tableofcontents{%
    \if@twocolumn\@restonecoltrue\onecolumn\else\@restonecolfalse\fi
    \par{\normalfont\Large\sffamily\bfseries\contentsname}\nopagebreak\par\bigskip
    \def\@chapter@code{}%
    \markboth{\contentsname}{\contentsname}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление алфавитного указателя авторов
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewenvironment{theindex}{%
    \if@twocolumn\@restonecolfalse\else\@restonecoltrue\fi
    \columnseprule \z@
    \columnsep 35\p@
    \twocolumn[{\normalfont\Large\sffamily\bfseries Авторский указатель}\vskip2em]%
    \markboth{Авторский указатель}{Авторский указатель}%
    \parindent\z@
    \parskip\z@ \@plus .3\p@\relax
    \flushright
    \let\item\@idxitem
}{%
    \if@restonecol\onecolumn\else\clearpage\fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление библиографии, в каждой статье отдельно
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Оформление элементов пункта библиографии
% INTERFACE
%\def\BibUrl#1{\\{\footnotesize\url{http://#1}}}
\def\BibAuthor#1{\emph{#1}}
\def\BibTitle#1{{#1}}
\def\BibJournal#1{\emph{#1}}
%\def\BibTitle#1{}  % Возможно, названия потом уберут везде
\def\BibUrl#1{{\small\url{#1}}}
\def\BibHttp#1{{\small\url{http://#1}}}
\def\BibFtp#1{{\small\url{ftp://#1}}}
\def\typeBibItem{\small\sloppy}

% Переопределение горизонтальных и вертикальных промежутков в списке литературы
\renewenvironment{thebibliography}[1]
    {\section{\bibname}%
        \list{\@biblabel{\@arabic\c@enumiv}}{%
            \settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin by 1ex%
            \topsep=0pt\parsep=3pt\itemsep=0ex%
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}%
        }%
        \typeBibItem
%        \clubpenalty4000%
%        \@clubpenalty\clubpenalty
%        \widowpenalty4000%
%        \sfcode`\.\@m%
    }{%
        \def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}%
        \endlist
    }

% генерация ссылки без квадратных скобок, полезна для самостоятельного оформления диапазонов вроде [X--Y,Z]
\def\citenb#1{%
    \let\temp@cite\@cite
    \begingroup
    \def\@cite##1##2{##1}%
    \cite{#1}%
    \endgroup
    \let\@cite\temp@cite
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Решение проблемы с конфликтом одноимённых меток, определённых в разных статьях
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Счётчик статей в сборнике для доопределения меток
\newcounter{PaperNo}
\def\thePaperNo{\arabic{PaperNo}}

% Доопределение меток в ссылках (добавление :\thePaperNo: в стандартные определения)
\let\globallabel\label
\let\globalltx@label\ltx@label
\let\globalref\ref
\let\globalpageref\pageref
\let\globalcitex\@citex
\let\globalbibitem\bibitem
\def\mmro@label#1{\globallabel{#1:\thePaperNo:}}
\def\mmro@ref#1{\globalref{#1:\thePaperNo:}}
\def\mmro@pageref#1{\globalpageref{#1:\thePaperNo:}}
\def\mmro@bibitem#1{\globalbibitem{#1:\thePaperNo:}}
\def\mmro@citex[#1]#2{%
  \let\@citea\@empty
  \@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{,\penalty\@m\ }%
     \edef\@citeb{\expandafter\@firstofone\@citeb\@empty:\thePaperNo:}%
     \if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
     \@ifundefined{b@\@citeb}{\mbox{\reset@font\bfseries ?}%
       \G@refundefinedtrue
       \@latex@warning{Citation `\@citeb' on page \thepage \space undefined}}%
       {\hbox{\csname b@\@citeb\endcsname}}}}{#1}%
}

% Переопределение команд работы с ссылками на локальные метки
% INTERFACE
\newcommand\SetPrivateLabeling{%
    \let\label\mmro@label
    \let\ltx@label\mmro@label % для корректной работы AmSLaTeX
    \let\ref\mmro@ref
    \let\pageref\mmro@pageref
    \let\@citex\mmro@citex
    \let\bibitem\mmro@bibitem
}

% При необходимости можно откатить изменения всех команд, работающих с метками
% INTERFACE
\newcommand\RestoreDefaultLabeling{
    \let\label\globallabel
    \let\ltx@label\globalltx@label
    \let\ref\globalref
    \let\pageref\globalpageref
    \let\@citex\globalcitex
    \let\bibitem\globalbibitem
}

% Действия, которые делаются в начале каждой статьи
% (как в сборнике, так и при отдельной компиляции)
\newcommand{\@BeginDocument}{
    % Переопределение списков с меньшими интервалами (слегка экономим бумагу)
    \renewcommand{\@listi}{%
        \topsep=\smallskipamount % вокруг списка
        \parsep=0pt% между абзацами внутри пункта
        \parskip=0pt% между абзацами
        \itemsep=0pt% между пунктами
        \itemindent=0pt% абзацный выступ
        \labelsep=1.5ex% расстояние до метки
        \leftmargin=3.5ex% отступ слева
        \rightmargin=0pt} % отступ справа
    \renewcommand{\@listii}{\@listi\topsep=0pt}%
    \renewcommand{\@listiii}{\@listii}%
    \renewcommand{\@listiv}{\@listii}%
    \renewcommand{\labelitemi}{---}%
    \renewcommand{\labelitemii}{---}%
    \renewcommand{\labelitemiii}{---}%
    \renewcommand{\labelitemiv}{---}%
    \renewcommand{\theenumii}{\asbuk{enumii}}%
    % Обнуление счётчиков
    \setcounter{equation}{0}%
    \setcounter{table}{0}%
    \setcounter{figure}{0}%
    \setcounter{algorithm}{0}%
    \setcounter{footnote}{0}%
    \setcounter{Theorem}{0}%
    \setcounter{State}{0}%
    \setcounter{Corollary}{0}%
    \setcounter{Def}{0}%
    \setcounter{Axiom}{0}%
    \setcounter{Hypothesis}{0}%
    \setcounter{Assumption}{0}%
    \setcounter{Problem}{0}%
    \setcounter{Example}{0}%
    \setcounter{Remark}{0}%
    \setcounter{Fact}{0}%
    \setcounter{Notice}{0}%
    \setcounter{Rule}{0}%
    \setcounter{Condition}{0}%
    % счётчик замечаний рецензента
    \setcounter{mmroReviewerNote}{0}%
    % счётчик статей
    \refstepcounter{PaperNo}%
    % русификация
    \ruseng{\hyphenation{иои ммро рффи}}\relax
    \def\bibname{\ruseng{Литература}{References}}%
    \def\figurename{\ruseng{Рис.}{Fig.}}%
    \def\tablename{\ruseng{Таблица}{Table}}%
}

% Действия, которые делаются в конце каждой статьи
% (только для сборника; при отдельной компиляции статьи всё это не нужно)
\newcommand\@EndDocument{%
    % единственное в этой команде, что может повлиять на текст:
    \balance
    % метка конца статьи
    \globallabel{\@paper@name:end}%
    % архивируются статьи, прошедшие корректуру и/или содержащие \REVIEWERNOTE
    \ifthenelse{\equal+{\@paper@corrected}\or\not\equal{\themmroReviewerNote}{0}}{%
        \ifwritebat
            \@write@corrauthors@bat{\ZipAdd \@paper@name \@paper@name.tex \@paper@name.pdf}%
            \@for\@pix@elem:=\@paper@pixlist\do{%
                \@write@corrauthors@bat{\ZipAdd \@paper@name \@pix@elem}%
            }%
        \fi
    }{}%
    % в оглавление вставляется отметка о количестве замечаний рецензента
    \ifthenelse{\equal{\themmroReviewerNote}{0}}{}{%
        \ifdraft{% если черновая печать
            \addtocontents{toc}{%
                \protect\par
                \protect\nopagebreak$\checkmark\kern-1.36ex\checkmark$\quad
                \textsl{В текст статьи включены замечания рецензента: \themmroReviewerNote}}%
        }\fi
    }%
    % пишется информация в текстовый файл для базы статей
    \ifwritetab
        \@write@papers@table{%
            \@paper@name|% имя файла
            \@chapter@code|% код раздела
            \@author|% авторы или список авторов через запятую
            \@title|% название статьи
            \@author@eng|% авторы или список авторов через запятую на английском
            \@title@eng|% название статьи на английском
            \@organization|% организация
            \@email|% адрес или список адресов, как их указали авторы
            \@reviewers@list|% список рецензентов и корректоров
            \@paper@corrected|% + прошла корректуру / - не прошла
            \@paper@reviewed|% + прошла рецензирование / - не прошла
            \@paper@accepted|% + принята в печать / ? желательна доработка / ! необходима доработка / - не принята
            \themmroReviewerNote % количество замечаний \REVIEWERNOTE, вставленных рецензентами в текст статьи
        }%
    \fi
    % пишется информация в HTML-файл для списка статей -- если выполняется условие включения статьи
    \ifthenelse{%
        \(\equal{\@paper@corrected}{\@@corrected}\or \equal{\@@corrected}{*}\) \and
        \(\equal{\@paper@reviewed}{\@@reviewed}\or \equal{\@@reviewed}{*}\) \and
        \(\equal{\@paper@accepted@}{\@@accepted}\or \equal{\@@accepted}{*}\)
    }{%
        \ifwritehtml
        \@write@papers@html{<br><i>\@author</i> <b>\@title</b>}%
        %\@write@papers@html{<br>({{\@organization}})}%
        \@write@papers@html{<br>{{\@abstract}}}%
        \fi
    }{}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Вставка статей, отслеживание файлов, сборка пакетов для авторов и рецензентов
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Вспомогательные файлы, генерируемые при компиляции сборника
\ifwritetab
    \newwrite\@papers@table
\fi
\ifwritehtml
    \newwrite\@papers@html
\fi
\ifwritebat
    \newwrite\@reviewers@bat
    \newwrite\@authors@bat
    \newwrite\@corrauthors@bat
    \newwrite\@make@bat
\fi
% Команда добавления файла в архив, записываемая в bat-файлы
% INTERFACE
\newcommand\ZipAdd[1]{rar a #1.rar }
%\newcommand\ZipAdd[1]{zzip -add #1.zip }

% Окружение для вставки статей целиком
\newenvironment{papers}{
    \gdef\@paper@name{}%
    \gdef\@paper@pixlist{}%
    \renewcommand{\documentclass}[2][0]{\relax}%
    \renewcommand{\usepackage}[2][0]{\relax}%
    \renewcommand\RestoreDefaultLabeling{\relax}%
    \ifdraft\relax\else\NOREVIEWERNOTES\fi
    \chaptercode{}
    \gdef\typeChapterCode##1{(##1)}
    \renewenvironment{document}{\SetPrivateLabeling}{\@EndDocument}%
    % создание файла со списком всех статей papers.tab
    \ifwritetab
        \immediate\openout\@papers@table=_papers.tab
        \gdef\@write@papers@table##1{%
            \immediate\write\@papers@table{##1}%
        }%
    \fi
    % создание файла со списком всех статей papers.html
    \ifwritehtml
        \immediate\openout\@papers@html=_papers.html
        \gdef\@write@papers@html##1{%
            \protected@write\@papers@html{}{##1}%
            %\immediate\write\@papers@html{##1}%
        }%
    \fi
    % создание файла reviewers.bat
    \ifwritebat
        \immediate\openout\@reviewers@bat=reviewers.bax
        \gdef\@write@reviewers@bat##1{%
            \immediate\write\@reviewers@bat{##1}%
        }%
        % создание файла authors.bat
        \immediate\openout\@authors@bat=authors.bax
        \gdef\@write@authors@bat##1{%
            \immediate\write\@authors@bat{##1}%
        }%
        % создание файла corr-authors.bat
        % архивируются только статьи, прошедшие корректуру и/или содержащие \REVIEWERNOTE
        \immediate\openout\@corrauthors@bat=corr-authors.bax
        \gdef\@write@corrauthors@bat##1{%
            \immediate\write\@corrauthors@bat{##1}%
        }%
        % создание файла make-separately.bat
        \immediate\openout\@make@bat=make-separately.bax
        \gdef\@write@make@bat##1{%
            \immediate\write\@make@bat{##1}%
        }%
        \@write@make@bat{del *.aux *.log *.toc *.dvi *.pdf}%
    \fi
    % если сборник, то в самом конце ничего не делать
    \AtEndDocument{\relax}
}{%
    \chaptercode{}
}

% если отдельная статья, то в~конце определить метку и сбалансировать колонки
\AtEndDocument{\globallabel{\@paper@name:end}\balance}

\def\@paper@name{\jobname}% имя текущей статьи
\def\@paper@pixlist{}% список картинок в статье
\def\@reviewers@list{}% список рецензентов и корректоров для текущей статьи
\def\@@reviewer{*}% рецензент, для которого собирается полный текст; по умолчанию все рецензенты

% подсчет числа статей
\newcounter{mmroTotal}
\newcounter{mmroCorrected}
\newcounter{mmroReviewed}
\newcounter{mmroAccepted}
\newcounter{mmroMayBeCompleted}
\newcounter{mmroMustBeCompleted}
\newcounter{mmroRejected}
\def\@paper@corrected{-}
\def\@paper@reviewed{-}
\def\@paper@accepted{+}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Команда включения статьи в сборник с отметками о текущем состоянии статьи:
% \paper123{автор}[рецензент]
%       1   ={+/-}  -статья прошла/нет корректуру
%        2  ={+/-}  -статья прошла/нет рецензирование
%         3 ={+/-/?/!}  -статья принята/отвергнута/может/должна быть доработана
% при включении статей полагается: ?=+  !=-
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Включать в сборник статьи только с этим набором отметок (* для всех)
% INTERFACE
\def\FILTER#1#2#3{%
    \gdef\@@corrected{#1}%
    \gdef\@@reviewed{#2}%
    \gdef\@@accepted{#3}%
    \@ifnextchar[\@FILTER@reviewer\relax%
}
\def\@FILTER@reviewer[#1]{\gdef\@@reviewer{#1}}
\ifdraft
    {\FILTER***}
\else
    {\FILTER**+}
\fi

% Команда для вставки статьи целиком
% INTERFACE
\def\paper#1#2#3#4{%
    \gdef\@paper@corrected{#1}
    \gdef\@paper@reviewed{#2}
    \gdef\@paper@accepted{#3}
    \gdef\@paper@name{#4}\refstepcounter{mmroTotal}%
    \gdef\@paper@pixlist{}
    \ifthenelse{\equal+{#1}}{\refstepcounter{mmroCorrected}}{}%
    \ifthenelse{\equal+{#2}}{\refstepcounter{mmroReviewed}}{}%
    \ifthenelse{\equal+{#3}}{\refstepcounter{mmroAccepted}}{}%
    \ifthenelse{\equal?{#3}}{\refstepcounter{mmroMayBeCompleted}}{}%
    \ifthenelse{\equal!{#3}}{\refstepcounter{mmroMustBeCompleted}}{}%
    \ifthenelse{\equal-{#3}}{\refstepcounter{mmroRejected}}{}%
    \ifwritebat
        \@write@authors@bat{\ZipAdd \@paper@name \@paper@name.tex \@paper@name.pdf}%
    \fi
    \@ifnextchar[\@paper@ii\@paper@i
}
\def\@paper@i{%
    \gdef\@reviewers@list{}%
    \ifthenelse{\equal{\@@reviewer}{}\or\equal{\@@reviewer}{*}}{\@paper}{}%
    \gdef\@paper@name{}%
}
\newboolean{bReviewer}
\def\@paper@ii[#1]{%
    \gdef\@reviewers@list{#1}%
    \setboolean{bReviewer}{false}%
    \@for\@reviewer@name:=\@reviewers@list\do{%
        \ifthenelse{\equal{\@@reviewer}{\@reviewer@name}}
            {\setboolean{bReviewer}{true}}%
            {}%
    }%
    \ifthenelse{\boolean{bReviewer}\or\equal{\@@reviewer}{*}}{%
        \@paper
        \@for\@reviewer@name:=\@reviewers@list\do{%
            \ifthenelse{\equal{\@@reviewer}{\@reviewer@name}\or\equal{\@@reviewer}{*}}
                {\ifwritebat
                    \@write@reviewers@bat{\ZipAdd \@reviewer@name iip9.sty \@paper@name.tex \@paper@name.pdf}%
                \fi}%
                {}%
        }%
    }{}%
    \gdef\@paper@name{}%
}
\def\GhostScriptExe{C:\Program Files (x86)\gs\gs8.64\bin\gswin32c.exe}
\def\@paper{%
    \begingroup
    \gdef\@paper@accepted@{\@paper@accepted}%
    \ifthenelse{\equal?{\@paper@accepted}}{\gdef\@paper@accepted@{+}}{}% всё-таки включаем
    \ifthenelse{\equal!{\@paper@accepted}}{\gdef\@paper@accepted@{-}}{}% пока не пришла доработка, не включаем
    \ifthenelse{%
        \(\equal{\@paper@corrected}{\@@corrected}\or \equal{\@@corrected}{*}\) \and
        \(\equal{\@paper@reviewed}{\@@reviewed}\or  \equal{\@@reviewed}{*}\) \and
        \(\equal{\@paper@accepted@}{\@@accepted}\or  \equal{\@@accepted}{*}\)
    }{%
        \ifwritebat
            \@write@make@bat{TeXify \@paper@name.tex}%
            %\@write@make@bat{DviPdfm -p a4 \@paper@name.dvi}%
            \@write@make@bat{dvips.exe \@paper@name.dvi}%  --- закомментировать при использовании DviPdfm
            \@write@make@bat{"\GhostScriptExe" -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sPAPERSIZE=a4 -r600 -dCompatibilityLevel=1.4 -sOutputFile="\@paper@name.pdf" -c save pop -f "\@paper@name.ps"}%  --- закомментировать при использовании DviPdfm
            \@write@make@bat{del \@paper@name.ps}%  --- закомментировать при использовании DviPdfm
        \fi
        \Russian
        \input{\@paper@name}%
    }%
    {}%
    \endgroup
}

% Занести имя включаемого графического файла в архивы и огравление
\newcommand\@include@graphics@file[1]{%
    % каждому рецензенту добавить данный графический файл в архив
    \@for\@reviewer@name:=\@reviewers@list\do{%
        \ifthenelse{\equal{\@@reviewer}{\@reviewer@name}\or\equal{\@@reviewer}{*}}{%
            \ifwritebat
                \@write@reviewers@bat{\ZipAdd \@reviewer@name #1}
            \fi
        }{}%
    }%
    % сформировать список всех включаемых графических файлов
    \ifthenelse{\equal{\@paper@name}{}}{}{%
        \ifwritebat
            \@write@authors@bat{\ZipAdd \@paper@name #1}%
        \fi
        \xdef\@old@paper@pixlist{\@paper@pixlist}%
        \ifthenelse{\equal{\@old@paper@pixlist}{}}%
            {\gdef\@paper@pixlist{#1}}%
            {\xdef\@paper@pixlist{\@old@paper@pixlist,#1}}%
    }%
%    % в черновом режиме в оглавление вставляются отметки о включённых графических файлах
%    \ifdraft%
%        \addtocontents{toc}{\protect\par\protect\nopagebreak$\blacksquare$\quad\protect\url{#1}}%
%    \fi
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Поддержка рецензирования и корректуры
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcounter{mmroReviewerNote}
\newcounter{mmroTotalReviewedPapers}
\newcounter{mmroTotalReviewerNotes}
\newcounter{mmroFinalReviewerNotes}

% Язык замечаний
\newcommand\LANGNOTE{\Rus}

% Проглотить следующий пробел
\def\gobblespace{\@ifnextchar\ {\hspace{-1ex}}\relax}

% Вставка замечания рецензента
\newcommand\typeREVIEWERNOTE[1]{{%
    \LANGNOTE\itshape\bfseries%\color{red}%
    \marginpar{%\raisebox{-1ex}{%\color{red}%
        $\checkmark\!_{\themmroReviewerNote}$%
    }%}%
    \{#1\}
}\gobblespace}
\newcommand\REVIEWERNOTE[1]{%
    \refstepcounter{mmroReviewerNote}%
    \refstepcounter{mmroTotalReviewerNotes}%
    \ifthenelse{\equal{\themmroReviewerNote}{1}}%
        {\refstepcounter{mmroTotalReviewedPapers}}%
        {}%
    \@type@REVIEWERNOTE{#1}%
}

% Вставка ответа автора на замечание
\newcommand\typeREPLY[1]{{\LANGNOTE\sffamily\itshape\bfseries\par---\;{#1}}}
\newcommand\REPLY[1]{\@type@REPLY{#1}}

% Счётчик авторов для генерации текста обращения
\newcounter{@authors@no}
\def\the@authors@no{\arabic{@authors@no}}

% Итоговое замечание в конце отрецензированной статьи
\newcommand\typeFINALREVIEWERNOTE{%
    \par\noindent
    \begin{minipage}{\columnwidth}\parindent=3.5ex
        \par\bigskip\hrule\vskip1pt\hrule\nopagebreak\bigskip
        \ruseng{Всего сделано замечаний}{Total reviewers' notes}: \themmroReviewerNote.
        \par\medskip
        % подсчёт числа авторов для генерации текста обращения
        \setcounter{@authors@no}{0}%
        \@for\@indx@elem:=\@author\do{%
            \addtocounter{@authors@no}{1}%
        }%
        \ifthenelse{\equal{\the@authors@no}{1}}%
            {\ruseng{Уважаемый автор!}{Dear author!}}%
            {\ruseng{Уважаемые авторы!}{Dear authors!}}%
        \par\medskip
        \ruseng{%
            Данная версия Вашей статьи может содержать редакторские правки.
            Если Вы собираетесь вносить свои правки, пожалуйста,
            редактируйте именно данную версию, а~не~Ваш исходный файл.
            \par
            При внесении исправлений желательно не пользоваться программами,
            которые самопроизвольно изменяют \LaTeX овскую разметку,
            так как это может испортить работу наших корректоров.
            Используйте текстовые редакторы TeXnicCenter, Texmaker, WinEdt, и~др.
            \par
            Рецензенты могли вставлять свои замечания прямо в~текст статьи
            с~помощью команды \texttt{\char"5C REVIEWERNOTE}.
            Убедительная просьба~--- не~удалять их из~документа.
            При печати сборника они не~будут видны.
            При необходимости Вы можете вставить ответ командой
            \mbox{\texttt{\char"5C REPLY\{}\textit{текст ответа}\texttt{\}}}
            Однако авторам рекомендуется использовать эту возможность лишь в~крайних случаях.
            Если рецензент понял Вас неправильно,
            то~это скорее повод улучшить изложение, чем вступать в~полемику с~рецензентом.
            Кроме того, нет гарантии, что рецензент сможет прочитать Ваш ответ.
            Авторам также рекомендуется внимательнее отнестись к~исправлениям
            и~не~вносить их второпях~--- это может только ухудшить статью.
            %\par
            Чтобы распечатать статью, скрыв замечания и~ответы,
            включите в~преамбулу команду
            \texttt{\char"5C NOREVIEWERNOTES}.%
        }{%
            This version of your paper contains a~blue pencil.
            If~you want to make your corrections,
            please edit this particular version, not the file you sent to us previously.
            \par
            Please, do not use software that modifies the \LaTeX\ markup automatically,
            because it can violate the output of our proofreaders.
            Text editors like TeXnicCenter, Texmaker, WinEdt, notepad can be used.
            \par
            Reviewers and proofreaders may insert notes directly in the text
            using the \texttt{\char"5C REVIEWERNOTE} command.
            Please, do~not delete these notes from the document.
            They will be hidden in the fair copy.
            If~necessary, you~can reply the note using the
            \mbox{\texttt{\char"5C REPLY\{}\textit{your text}\texttt{\}}}
            command.
            %\par
            To~print the paper without notes and replies
            insert the command \texttt{\char"5C NOREVIEWERNOTES}
            in~the preamble of the document.%
        }%
        \par\nopagebreak\medskip\hrule\nopagebreak\bigskip
    \end{minipage}
}
\newcommand\FINALREVIEWERNOTE{%
    \refstepcounter{mmroFinalReviewerNotes}%
    \ifthenelse{\equal{\@paper@name}{\jobname}}{% если печать отдельного файла
        \@type@FINALREVIEWERNOTE
    }{}%
}

% Последний срок отправки доработанной версии
\newcommand\typeDEADLINE{\ruseng{1~июля 2012~года}{July~1, 2012}}

% Текст предупреждения об автогенерируемом тексте
\newcommand\typeAUTOWARN[1]{%
    \noindent\vskip-1.7ex
    {%\centering%\footnotesize
        \ruseng{%
            Весь последующий текст выводится автоматически,
            согласно одному из трёх возможных решений Программного комитета:
            принять, отдать~на~доработку, либо отвергнуть.
        }{%
            Further text was generated automatically
            according to one of three possible decisions of the Program Committee:
            accept, amend, or reject.
        }%
    \par
    Данная статья \textbf{#1}.
    \par\smallskip\hrule}%
    \bigskip
}

% Три типа итоговых замечаний с~окончательным решением о принятии/доработке/отказе
\newcommand\typeACCEPTNOTE{\par
    %\typeAUTOWARN{принята}%
    {\itshape\bfseries
        \LANGNOTE\ruseng{%
            Статья прошла рецензирование, корректуру и~принята к~публикации в~сборнике.
            Вы~можете подать доработанную версию до \typeDEADLINE\ включительно (чем раньше, тем лучше).
            Если оргкомитет не получит от Вас доработанную версию до указанного срока,
            то~будет опубликована данная версия.
        }{%
            The paper has been passed the phases of reviewing and proofreading.
            The paper is accepted.
            If~necessary You can make changes and submit your final version
            before \typeDEADLINE\ inclusively (the earlier the better).
            If~You submit nothing this version will be published.
        }%
    }\par
}
\newcommand\typeAMENDNOTE{\par
    %\typeAUTOWARN{отдана на~доработку}%
    {\itshape\bfseries
        \LANGNOTE\ruseng{%
            Статья может быть принята к~публикации в~сборнике только при условии её существенной доработки.
            Вы~можете подать доработанную версию до \typeDEADLINE\ включительно (чем раньше, тем лучше).
            Если оргкомитет не получит от Вас доработанную версию до указанного срока,
            то~статья не будет опубликована.
        }{%
            The paper has been passed the phases of reviewing and proofreading.
            The paper may be accepted only after Your amendment (see reviewers' notes in the text).
            You can submit your final version before \typeDEADLINE\ inclusively (the earlier the better).
            If~You submit nothing the paper will not be published.
        }%
    }\par
}
\newcommand\typeREJECTNOTE{\par
    %\typeAUTOWARN{отвергнута}%
    {\itshape\bfseries
        \LANGNOTE\ruseng{%
            Статья не~может быть принята к~публикации в~сборнике.
            Вы~можете подать полностью переработанную версию статьи до \typeDEADLINE\ включительно (чем раньше, тем лучше).
            Если оргкомитет не получит от Вас новую версию до указанного срока,
            то~статья не будет опубликована.
        }{%
            The paper is rejected.
            Nevertheless, You can make changes and submit your final version
            before \typeDEADLINE\ inclusively (the earlier the better).
            If~You submit nothing the paper will not be published.
        }%
    }\par
}

\newcommand\ACCEPTNOTE{%
    \refstepcounter{mmroFinalReviewerNotes}%
    \ifthenelse{\equal{\@paper@name}{\jobname}}{% если печать отдельного файла
        \@type@ACCEPTNOTE
    }{}%
}
\newcommand\AMENDNOTE{%
    \refstepcounter{mmroFinalReviewerNotes}%
    \ifthenelse{\equal{\@paper@name}{\jobname}}{% если печать отдельного файла
        \@type@AMENDNOTE
    }{}%
}
\newcommand\REJECTNOTE{%
    \refstepcounter{mmroFinalReviewerNotes}%
    \ifthenelse{\equal{\@paper@name}{\jobname}}{% если печать отдельного файла
        \@type@REJECTNOTE
    }{}%
}

\newcommand\@type@REVIEWERNOTE[1]{\typeREVIEWERNOTE{#1}}
\newcommand\@type@REPLY[1]{\typeREPLY{#1}}
\newcommand\@type@FINALREVIEWERNOTE{\typeFINALREVIEWERNOTE}
\newcommand\@type@ACCEPTNOTE{\par\bigskip\hrule\nopagebreak\bigskip\typeACCEPTNOTE\typeFINALREVIEWERNOTE}
\newcommand\@type@AMENDNOTE{\par\bigskip\hrule\nopagebreak\bigskip\typeAMENDNOTE\typeFINALREVIEWERNOTE}
\newcommand\@type@REJECTNOTE{\par\bigskip\hrule\nopagebreak\bigskip\typeREJECTNOTE\typeFINALREVIEWERNOTE}

% Отмена вывода всех рецензентских замечаний
\newcommand\NOREVIEWERNOTES{%
    \renewcommand\@type@REVIEWERNOTE[1]{\gobblespace}
    \renewcommand\@type@REPLY[1]{\gobblespace}
    \renewcommand\@type@FINALREVIEWERNOTE{\gobblespace}
    \renewcommand\@type@ACCEPTNOTE{\gobblespace}
    \renewcommand\@type@AMENDNOTE{\gobblespace}
    \renewcommand\@type@REJECTNOTE{\gobblespace}
}

% Полная статистика по всему сборнику
\newcommand\@type@FINALSTAT{%
    \newpage\chaptercode{}\Russian
    %\bigskip\hrule\vskip1pt\hrule height2pt\bigskip
    {\normalfont\Large\sffamily\bfseries Итоговая статистика статей}
    \vskip2em
    \par
    Всего подано статей: \themmroTotal.
    \par
    Статей, прошедших корректуру: \themmroCorrected.
    \\\par
    Статей, прошедших рецензирование: \themmroReviewed.
    \par
    Всего статей c \texttt{\char"5C REVIEWERNOTE}: \themmroTotalReviewedPapers\
    (всего сделано замечаний: \themmroTotalReviewerNotes).
    \par
    Всего статей с \texttt{\char"5C FINALREVIEWERNOTE}: \themmroFinalReviewerNotes.
    \\\par
    Статей, отмеченных как принятых в~печать: \themmroAccepted.
    \par
    Статей, которые желательно доработать: \themmroMayBeCompleted.
    \par
    Статей, которые не могут быть опубликованы без доработки: \themmroMustBeCompleted.
    \par
    Статей, которые уже точно отброшены: \themmroRejected.
    \\\par
    Статей в~этой версии сборника: \thePaperNo.
    \par
    Всего иллюстраций в~этой версии сборника: \thePictureNo.
}
% INTERFACE
\newcommand\FINALSTAT{%
    \ifdraft\@type@FINALSTAT\fi
}

\newcounter{TodoCount}
\def\thevkTodoCount{\arabic{TodoCount}}
% INTERFACE
\newcommand\TODO[1]{{%
    \small\color{red}%
    \marginpar{\raisebox{-1ex}{\color{red}ToDo$^{\refstepcounter{TodoCount}\theTodoCount}$}}%
    \{#1\}
}}
%\renewcommand\TODO[1]{\gobblespace}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Для включения графиков пакетом graphicx
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareGraphicsRule{.wmf}{bmp}{}{}
\DeclareGraphicsRule{.emf}{bmp}{}{}
\DeclareGraphicsRule{.bmp}{bmp}{}{}
\DeclareGraphicsRule{.png}{bmp}{}{}
% Для подписей на рисунках, вставляемых includegraphics
\def\XYtext(#1,#2)#3{\rlap{\kern#1\lower-#2\hbox{#3}}}

% Переопределение вставки графики
\newcounter{PictureNo}
\let\IncludeGraphics=\includegraphics
\renewcommand\includegraphics[2][]{%
    \IncludeGraphics[#1]{#2}%
    \refstepcounter{PictureNo}%
    % если идёт вёрстка сборника, включить картинку в архивы
    \ifthenelse{\equal{\@paper@name}{\jobname}}%
        {}%
        {\@include@graphics@file{#2}}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Плавающие иллюстрации
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\setcounter{topnumber}{9}
\setcounter{totalnumber}{9}
\renewcommand\topfraction{1.0}
\renewcommand\textfraction{0.0}
\renewcommand\floatpagefraction{0.01} % float-страниц быть вообще не должно - это чтобы их лучше видеть ;)
\setlength\floatsep{2ex}
\setlength\textfloatsep{2.5ex}
\setlength\intextsep{2.5ex}
\setlength\abovecaptionskip{2ex}

%\def\@caption@left@right@skip{\leftskip=3.5ex\rightskip=3.5ex}
\def\@caption@left@right@skip{\leftskip=0ex\rightskip=0ex}
\def\nocaptionskips{\def\@caption@left@right@skip{}}
\def\fnum@figure{\figurename\:\thefigure} %% нерастяжимый тонкий пробел между Рис. и номером рисунка
\def\fnum@table{\tablename\enskip\thetable} %% нерастяжимый тонкий пробел между Таблица и номером таблицы

\renewcommand\@makecaption[2]{%
    \vskip\abovecaptionskip
    \sbox\@tempboxa{\small\textbf{#1.} #2}%
    \ifdim\wd\@tempboxa >\hsize
        {\@caption@left@right@skip\small\textbf{#1.} #2\par}%
    \else
        \global\@minipagefalse
        \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
    \fi
    %\vskip\belowcaptionskip
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Некоторые переопределения для унификации математики
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand{\geq}{\geqslant}
\renewcommand{\leq}{\leqslant}
\renewcommand{\ge}{\geqslant}
\renewcommand{\le}{\leqslant}
\renewcommand{\emptyset}{\varnothing}
\renewcommand{\kappa}{\varkappa}
\let\oldphi=\phi
\renewcommand{\phi}{\varphi}
\renewcommand{\epsilon}{\varepsilon}

\let\varvec\vec
%\renewcommand{\vec}[1]{\mathbf{#1}}
\renewcommand{\vec}[1]{\boldsymbol{#1}}
%\def\ubar#1{\smash[b]{\underset{\rule[0.8ex]{3pt}{0.3pt}}#1}}
\renewcommand{\complement}{\mathsf{C}}
\newcommand{\T}{^{\text{\tiny\sffamily\upshape\mdseries T}}}
%\newcommand{\T}{^{\textsf{\upshape т}}}
%\newcommand{\T}{^{\text{\rm т}}}

\newcommand\myop[1]{\mathop{\operator@font #1}\nolimits}
\newcommand\mylim[1]{\mathop{\operator@font #1}\limits}

%\DeclareMathSymbol{\sumop}{\mathop}{largesymbols}{"50}
%\def\sum{\sumop\limits}
\newcommand{\tsum}{\mathop{\textstyle\sum}\limits}
\newcommand{\tprod}{\mathop{\textstyle\prod}\limits}
\newcommand{\tvee}{\mathop{\textstyle\bigvee}\limits}
\newcommand{\twedge}{\mathop{\textstyle\bigwedge}\limits}
\newcommand{\tbigcap}{\mathop{\textstyle\bigcap}\limits}
\newcommand{\tbigcup}{\mathop{\textstyle\bigcup}\limits}
\renewcommand\lim{\mylim{lim}}
\renewcommand\limsup{\mylim{lim\,sup}}
\renewcommand\liminf{\mylim{lim\,inf}}
\renewcommand\max{\mylim{max}}
\renewcommand\min{\mylim{min}}
\renewcommand\sup{\mylim{sup}}
\renewcommand\inf{\mylim{inf}}
\newcommand\argmin{\mylim{arg\,min}}
\newcommand\argmax{\mylim{arg\,max}}
\newcommand\Tr{\myop{tr}}
\newcommand\rank{\myop{rank}}
\newcommand\diag{\myop{diag}}
\renewcommand\Re{\myop{Re}}
\renewcommand\Im{\myop{Im}}
\newcommand\sign{\mylim{sign}}
\newcommand\const{\myop{const}}

% теория вероятностей
\newcommand{\erf}{\myop{erf}}
\newcommand{\Expect}{\mathsf{E}}
\newcommand{\Var}{\mathsf{D}}
\newcommand\cov{\myop{cov}}
\newcommand\corr{\myop{corr}}
\newcommand\Normal{\mathcal{N}}
\newcommand{\cond}{\mspace{3mu}{|}\mspace{3mu}}

% теория вычислительной сложности
% шрифт для классов сложности
%\def\CCfont#1{\ifmmode{\mathsf{#1}}\else\mbox{\textsf{#1}}\fi}
\def\CCfont#1{\ifmmode\mbox{\textsf{#1}}\else\mbox{\rm\textsf{#1}}\fi}
% шрифт для названий задач
\def\CPfont#1{\ifmmode\mbox{\textsc{#1}}\else\mbox{\rm\textsc{#1}}\fi}
% классы сложности
\def\P{\CCfont{P}}
\def\NP{\CCfont{NP}}
\def\DTIME{\CCfont{DTIME}}
\def\MaxSNP{\CCfont{Max-SNP}}
\def\Apx{\CCfont{Apx}}
% задачи
\def\PC{\CPfont{PC}}
\def\MinPC{\CPfont{MinPC}}
\def\threeSAT{\CPfont{3SAT}}
\def\GapSAT{\CPfont{Gap-3SAT}}

% множества
\def\NN{\mathbb{N}}
\def\ZZ{\mathbb{Z}}
\def\QQ{\mathbb{Q}}
\def\RR{\mathbb{R}}
\def\CC{\mathbb{C}}
\def\HH{\mathbb{H}}
%\def\LL{\mathbb{L}}
%\def\II{\mathbb{I}}
%\def\DD{\mathbb{D}}
%\def\cL{\mathscr{L}}
%\def\cF{\mathscr{F}}
%\def\cG{\mathscr{G}}
%\def\cB{\mathscr{B}}
%\def\cK{\mathscr{K}}
%\def\cJ{\mathcal{J}}
%\def\cN{\mathcal{N}}
%\def\fF{\mathfrak{F}}
%\def\fI{\mathfrak{I}}
%\def\fM{\mathfrak{M}}
%\def\fR{\mathfrak{R}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Перечни
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Нумерованный перечень со скобками
\def\afterlabel#1{\renewcommand\labelenumi{\theenumi #1}}
\def\aroundlabel#1#2{\renewcommand\labelenumi{#1\theenumi #2}}
\newenvironment{enumerate*}
{%
    \begingroup
    \renewcommand{\@listi}{%
        \topsep=\smallskipamount % вокруг списка
        \parsep=0pt% между абзацами внутри пункта
        \parskip=0pt% между абзацами
        \itemsep=0pt% между пунктами
        \itemindent=0ex% абзацный выступ
        \labelsep=1.5ex% расстояние до метки
        \leftmargin=7ex% отступ слева
        \rightmargin=0pt} % отступ справа
    \begin{enumerate}%
    \afterlabel)%
}{%
    \end{enumerate}%
    \endgroup
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Новые теоремоподобные окружения
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\theoremstyle{plain}
% Шаманские притоптывания, чтобы ставить точку после номера теоремы
\gdef\th@plain{\normalfont
    \def\@begintheorem##1##2{%
        \item[\hskip\labelsep\theorem@headerfont ##1\ ##2. ]}%
    \def\@opargbegintheorem##1##2##3{%
        \item[\hskip\labelsep\theorem@headerfont ##1\ ##2] {\theorem@headerfont(##3). }}%
}
% Теоремы наклонным шрифтом
\theorempreskipamount=\smallskipamount
\theorempreskipamount=\smallskipamount
\theorembodyfont{\rmfamily\slshape}
\newtheorem{Theorem}{\ruseng{Теорема}{Theorem}}
\newtheorem{Lemma}[Theorem]{\ruseng{Лемма}{Lemma}}
\newtheorem{State}{\ruseng{Утверждение}{Statement}}
\newtheorem{Corollary}{\ruseng{Следствие}{Corollary}}
\newtheorem{Def}{\ruseng{Определение}{Definition}}
\newtheorem{Definition}[Def]{\ruseng{Определение}{Definition}}
% Теоремы прямым шрифтом
\theorembodyfont{\rmfamily}
\newtheorem{Axiom}{\ruseng{Аксиома}{Axiom}}
\newtheorem{Hypothesis}{\ruseng{Гипотеза}{Hypothesis}}
\newtheorem{Assumption}{\ruseng{Предположение}{Assumption}}
\newtheorem{Problem}{\ruseng{Задача}{Problem}}
\newtheorem{Example}{\ruseng{Пример}{Example}}
\newtheorem{Fact}{\ruseng{Факт}{Fact}}
\newtheorem{Remark}{\ruseng{Замечание}{Remark}}
\newtheorem{Notice}{\ruseng{Замечание}{Notice}}
\newtheorem{Rule}{\ruseng{Правило}{Rule}}
\newtheorem{Condition}{\ruseng{Условие}{Condition}}
% Для доказательств (не рекомендуется в коротких статьях)
\newenvironment{Proof}%
    {\par\noindent{\bf Доказательство.}}%
    {\hfill$\scriptstyle\blacksquare$}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Оформление алгоритмов в пакетах algorithm, algorithmic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% переопределения (русификация) управляющих конструкций:
\newcommand{\algKeyword}[1]{{\bf #1}}
\renewcommand{\algorithmicrequire}{\rule{0pt}{2.5ex}\algKeyword{\ruseng{Вход:}{Require:}}}
\renewcommand{\algorithmicensure}{\algKeyword{\ruseng{Выход:}{Ensure:}}}
\renewcommand{\algorithmicend}{\algKeyword{\ruseng{конец}{end}}}
\renewcommand{\algorithmicif}{\algKeyword{\ruseng{если}{if}}}
\renewcommand{\algorithmicthen}{\algKeyword{\ruseng{то}{then}}}
\renewcommand{\algorithmicelse}{\algKeyword{\ruseng{иначе}{else}}}
\renewcommand{\algorithmicelsif}{\algorithmicelse\ \algorithmicif}
\renewcommand{\algorithmicendif}{\algorithmicend\ \algorithmicif}
\renewcommand{\algorithmicfor}{\algKeyword{\ruseng{для}{for}}}
\renewcommand{\algorithmicforall}{\algKeyword{\ruseng{для всех}{for all}}}
\renewcommand{\algorithmicdo}{}
\renewcommand{\algorithmicendfor}{\algorithmicend\ \algorithmicfor}
\renewcommand{\algorithmicwhile}{\algKeyword{\ruseng{пока}{while}}}
\renewcommand{\algorithmicendwhile}{\algorithmicend\ \algorithmicwhile}
\renewcommand{\algorithmicloop}{\algKeyword{\ruseng{цикл}{loop}}}
\renewcommand{\algorithmicendloop}{\algorithmicend\ \algorithmicloop}
\renewcommand{\algorithmicrepeat}{\algKeyword{\ruseng{повторять}{repeat}}}
\renewcommand{\algorithmicuntil}{\algKeyword{\ruseng{пока}{until}}}
%\renewcommand{\algorithmiccomment}[1]{{\footnotesize // #1}}
\renewcommand{\algorithmiccomment}[1]{{\quad\sl // #1}}

% Мои дополнительные команды для описания алгоритмов
\newcommand{\Procedure}[1]{{\tt #1}}
\newcommand{\Proc}[1]{\text{\tt #1}}
\def\BEGIN{\\[1ex]\hrule\vskip 1ex}
\def\PARAMS{\renewcommand{\algorithmicrequire}{\algKeyword{\ruseng{Параметры:}{Parameters:}}}\REQUIRE}
\def\END{\vskip 1ex\hrule\vskip 1ex}
\def\RET{\algKeyword{\ruseng{вернуть}{return}} }
\def\EXIT{\algKeyword{\ruseng{выход}{exit}}}
\def\IFTHEN#1{\STATE\algorithmicif\ #1 {\algorithmicthen}}
\def\PROCEDURE#1{\medskip\STATE\algKeyword{\ruseng{ПРОЦЕДУРА}{PROCEDURE}} \Procedure{#1}}

% Ещё несколько переопределений для алгоритмов
\renewcommand{\listalgorithmname}{Список алгоритмов}
\floatname{algorithm}{\ruseng{Алгоритм}{Algorithm}}
\floatplacement{algorithm}{!t}

% чтобы поставить точечку после номера алгоритма в \caption:
\renewcommand\floatc@ruled[2]{\vskip2pt\textbf{#1.} #2\par}

% чтобы можно было ссылаться на шаги алгоритма
\newenvironment{Algorithm}[1][t]%
    {\begin{algorithm}[#1]\begin{algorithmic}[1]%
        \renewcommand{\ALC@it}{%
            \refstepcounter{ALC@line}% удивительно, почему это не сделал Peter Williams?
            \addtocounter{ALC@rem}{1}%
            \ifthenelse{\equal{\arabic{ALC@rem}}{1}}{\setcounter{ALC@rem}{0}}{}%
            \item}}%
    {\end{algorithmic}\end{algorithm}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Рисование нейронных сетей и диаграмм
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{network}%
    {\catcode`"=12\begin{xy}<1ex,0ex>:}%
    {\end{xy}\catcode`"=13}
\def\nnNode"#1"(#2)#3{\POS(#2)*#3="#1"}
\def\nnLink"#1,#2"#3{\POS"#1"\ar #3 "#2"}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   П Р О Ч Е Е
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Для таблиц: невидимая вертикальная линейка немного расширяет заголовок
\newcommand{\hstrut}{\rule{0pt}{2.5ex}}
\newcommand{\headline}{\hline\hstrut}

% Выделение - всегда курсивом
\renewcommand\emph[1]{\textit{#1}}

% Перенос знака операции на следующую строку
\newcommand\brop[1]{#1\discretionary{}{\hbox{$\mathsurround=0pt #1$}}{}}

\begingroup
\catcode`\+\active\gdef+{\mathchar8235\nobreak\discretionary{}%
 {\usefont{OT1}{cmr}{m}{n}\char43}{}}
\catcode`\-\active\gdef-{\mathchar8704\nobreak\discretionary{}%
 {\usefont{OMS}{cmsy}{m}{n}\char0}{}}
\catcode`\=\active\gdef={\mathchar12349\nobreak\discretionary{}%
 {\usefont{OT1}{cmr}{m}{n}\char61}{}}
\endgroup
\def\cdot{\mathchar8705\nobreak\discretionary{}%
 {\usefont{OMS}{cmsy}{m}{n}\char1}{}}
\def\times{\mathchar8706\nobreak\discretionary{}%
 {\usefont{OMS}{cmsy}{m}{n}\char2}{}}
\AtBeginDocument{%
\mathcode`\==32768
\mathcode`\+=32768
\mathcode`\-=32768
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Для финальной печати убрать замечания рецензентов
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%\NOREVIEWERNOTES
